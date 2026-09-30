"""Follow-up call planning from OpenAPI Link Objects (``POST /links/plan``).

Given an OpenAPI 3.x document, a source ``METHOD /path`` operation selector,
a concrete request/response runtime context (the same envelope shape used by
``/callbacks/resolve``) and a Link name, locate the source operation's
response selected by the runtime status code (exact code, then the
``nXX`` range code, then ``default``), resolve the named Link Object and
produce the plan for the follow-up HTTP call:

* the target operation, addressed either by a document-local
  ``operationRef`` fragment or by the globally unique ``operationId``
  (exactly one of the two must be present);
* parameter values grouped into ``path`` / ``query`` / ``header`` /
  ``cookie`` after evaluating the Link ``parameters`` runtime expressions,
  matched against the target operation's parameter *names*;
* the follow-up ``requestBody`` after evaluating the Link
  ``requestBody`` runtime expression;
* the source JSON Pointer of every mapping.

Response, Link and target Path Item local ``$ref`` chains (including
OpenAPI 3.1 sibling fields) are resolved cycle-safely. Missing expression
values, invalid targets/parameters, unset required parameters and
external / dangling / circular references are all collected and rejected
together with HTTP 422; the request document and context are never
modified and no partial plan is returned.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from .materializer import _resolve_path_item  # type: ignore
from .refresolver import (
    MAX_REF_CHAIN,
    RefResolver,
    UnresolvableReferenceError,
    is_local_ref,
    parse_local_ref,
)
from .slicer import HTTP_METHODS, parse_selector

GROUPS = ("path", "query", "header", "cookie")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class LinkError(Exception):
    """Base class for link planning failures."""


class InvalidDocumentError(LinkError):
    """The submitted document is not a recognizable OpenAPI 3.x doc (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class InvalidSelectorError(LinkError):
    """The source selector is malformed or unknown (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class InvalidLinkRequestError(LinkError):
    """The request envelope (context/statusCode) is malformed (400)."""

    def __init__(self, messages: List[str]) -> None:
        self.messages = messages
        super().__init__("; ".join(messages))


class LinkPlanningError(LinkError):
    """One or more semantic link-planning problems (HTTP 422)."""

    def __init__(self, errors: List[Dict[str, Any]]) -> None:
        self.errors = errors
        super().__init__(
            "; ".join(f"{e['code']} @ {e.get('pointer', '')}" for e in errors)
        )


# --------------------------------------------------------------------------- #
# JSON Pointer helpers (RFC 6901)
# --------------------------------------------------------------------------- #


def escape_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def pointer_join(base: str, token: Any) -> str:
    return base + "/" + escape_token(str(token))


def unescape_token(token: str) -> str:
    # Order matters: "~1" must be expanded before "~0".
    return token.replace("~1", "/").replace("~0", "~")


def _valid_array_index(token: str, length: int) -> bool:
    if token == "0":
        return length > 0
    if not token.isdigit() or token[0] == "0":
        return False
    return int(token) < length


# --------------------------------------------------------------------------- #
# Runtime expressions (OpenAPI 3 specification)
# --------------------------------------------------------------------------- #

# RFC 7230 tchar (header field name characters).
_TCHAR = frozenset("!#$%&'*+-.^_`|~")
_NAME_PUNCT = frozenset("/?!$&'()*+,;=:@")
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)

# Descriptor shapes (same convention as app.callbacks):
#   ("simple", "url" | "method" | "status_code")
#   ("message", side, "path" | "query", raw_name)
#   ("message", side, "header", token)
#   ("message", side, "body", json_pointer | None)
Descriptor = Tuple[Any, ...]


def _is_hex_digit(char: str) -> bool:
    return char in "0123456789abcdefABCDEF"


def _valid_name(text: str) -> bool:
    index = 0
    while index < len(text):
        char = text[index]
        if char in _UNRESERVED or char in _NAME_PUNCT:
            index += 1
        elif (
            char == "%"
            and index + 2 < len(text)
            and _is_hex_digit(text[index + 1])
            and _is_hex_digit(text[index + 2])
        ):
            index += 3
        else:
            return False
    return True


def _valid_pointer_syntax(pointer: str) -> bool:
    if pointer == "":
        return True
    if not pointer.startswith("/"):
        return False
    for raw_token in pointer.split("/")[1:]:
        index = 0
        while index < len(raw_token):
            if raw_token[index] == "~":
                if index + 1 >= len(raw_token) or raw_token[index + 1] not in "01":
                    return False
                index += 2
            else:
                index += 1
    return True


def parse_link_expression(text: Any) -> Optional[Descriptor]:
    """Parse one complete Link runtime expression.

    Unlike callback URL templates, Link ``parameters`` / ``requestBody``
    values are always a single complete expression; embedded ``{...}``
    templates and literal strings are rejected. Structured expression
    results (objects/arrays) are allowed: only their absence is an error.
    """
    if not isinstance(text, str) or not text:
        return None
    if "{" in text or "}" in text:
        return None

    if text == "$url":
        return ("simple", "url")
    if text == "$method":
        return ("simple", "method")
    if text == "$statusCode":
        return ("simple", "status_code")

    for side in ("request", "response"):
        prefix = "$" + side + "."
        if not text.startswith(prefix):
            continue
        rest = text[len(prefix):]
        if not rest:
            return None

        if rest.startswith("header."):
            token = rest[len("header."):]
            if not token or not all(
                ch.isalnum() or ch in _TCHAR for ch in token
            ):
                return None
            return ("message", side, "header", token)

        if side == "request" and rest.startswith("query."):
            name = rest[len("query."):]
            if not name or not _valid_name(name):
                return None
            return ("message", side, "query", name)

        if side == "request" and rest.startswith("path."):
            name = rest[len("path."):]
            if not name or not _valid_name(name):
                return None
            return ("message", side, "path", name)

        if rest == "body":
            return ("message", side, "body", None)
        if rest.startswith("body#"):
            pointer = rest[len("body#"):]
            if not _valid_pointer_syntax(pointer):
                return None
            return ("message", side, "body", pointer)

        return None

    return None


class _Missing:
    """Sentinel for a value that could not be located in the context."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging only
        return "<MISSING>"


MISSING = _Missing()


def _navigate_body(body: Any, pointer: Optional[str]) -> Any:
    node = body
    if pointer:
        for raw_token in pointer.split("/")[1:]:
            token = unescape_token(raw_token)
            if isinstance(node, dict):
                if token not in node:
                    return MISSING
                node = node[token]
            elif isinstance(node, list):
                if not _valid_array_index(token, len(node)):
                    return MISSING
                node = node[int(token)]
            else:
                return MISSING
    return node


def _header_lookup(
    headers: Dict[str, Any], name: str
) -> Tuple[str, Optional[str], Any]:
    """Case-insensitive header lookup: missing / ambiguous / found."""
    lowered = name.lower()
    matches = [
        key for key in headers if isinstance(key, str) and key.lower() == lowered
    ]
    if not matches:
        return "missing", None, None
    if len(matches) > 1:
        return "ambiguous", None, None
    return "found", matches[0], headers[matches[0]]


def evaluate_expression(
    descriptor: Descriptor, context: Dict[str, Any]
) -> Tuple[str, Any, str]:
    """Evaluate a parsed descriptor against the runtime context.

    Returns ``(status, value, context_pointer)`` where status is one of
    ``"ok"`` / ``"missing"`` / ``"ambiguous"``. Structured values
    (objects/arrays) and ``null`` explicitly present in the body are
    returned as-is; a located JSON ``null`` is a legitimate value while a
    missing path is not.
    """
    kind = descriptor[0]
    if kind == "simple":
        name = descriptor[1]
        if name == "status_code":
            response = context.get("response")
            if not isinstance(response, dict) or "statusCode" not in response:
                return "missing", None, "/context/response/statusCode"
            return "ok", response["statusCode"], "/context/response/statusCode"

        request = context.get("request")
        field = "url" if name == "url" else "method"
        if not isinstance(request, dict) or field not in request:
            return "missing", None, f"/context/request/{field}"
        return "ok", request[field], f"/context/request/{field}"

    _, side, source, token = descriptor
    message = context.get(side)
    base_pointer = f"/context/{side}"

    if source in ("path", "query"):
        if not isinstance(message, dict) or source not in message:
            return "missing", None, f"{base_pointer}/{source}"
        values = message[source]
        if not isinstance(values, dict):
            return "missing", None, f"{base_pointer}/{source}"
        name = unquote(token)
        if name not in values:
            return (
                "missing",
                None,
                pointer_join(f"{base_pointer}/{source}", name),
            )
        return "ok", values[name], pointer_join(f"{base_pointer}/{source}", name)

    if source == "header":
        if not isinstance(message, dict) or "header" not in message:
            return "missing", None, f"{base_pointer}/header"
        headers = message["header"]
        if not isinstance(headers, dict):
            return "missing", None, f"{base_pointer}/header"
        status, matched_key, value = _header_lookup(headers, token)
        if status == "missing":
            return "missing", None, pointer_join(f"{base_pointer}/header", token)
        if status == "ambiguous":
            return (
                "ambiguous",
                None,
                pointer_join(f"{base_pointer}/header", token),
            )
        return (
            "ok",
            value,
            pointer_join(f"{base_pointer}/header", matched_key),
        )

    # body
    pointer = token
    if not isinstance(message, dict) or "body" not in message:
        return "missing", None, f"{base_pointer}/body"
    value = _navigate_body(message["body"], pointer)
    context_pointer = f"{base_pointer}/body" + (pointer or "")
    if value is MISSING:
        return "missing", None, context_pointer
    return "ok", value, context_pointer


# --------------------------------------------------------------------------- #
# Context envelope
# --------------------------------------------------------------------------- #

_REQUEST_FIELDS = ("url", "method", "path", "query", "header", "body")
_RESPONSE_FIELDS = ("statusCode", "header", "body")


def normalize_context(context: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Validate the request/response runtime context envelope.

    Mirrors the /callbacks envelope, but header values may be any JSON
    value because Link expressions may legitimately extract structured
    data (only URL interpolation requires scalars).
    """
    messages: List[str] = []
    if not isinstance(context, dict):
        return {}, ["context must be a JSON object"]

    for section in sorted(context, key=str):
        if section not in ("request", "response"):
            messages.append(
                f"context.{section}: unknown context section; expected one of "
                "'request'/'response'"
            )

    normalized: Dict[str, Any] = {}

    request = context.get("request")
    if request is not None:
        if not isinstance(request, dict):
            messages.append("context.request must be a JSON object")
        else:
            for field in sorted(request):
                if field not in _REQUEST_FIELDS:
                    messages.append(
                        f"context.request.{field}: unknown request field; "
                        "expected one of url/method/path/query/header/body"
                    )
            norm_request: Dict[str, Any] = {}
            url = request.get("url")
            if url is not None:
                if isinstance(url, str):
                    norm_request["url"] = url
                else:
                    messages.append("context.request.url must be a string")
            method = request.get("method")
            if method is not None:
                if isinstance(method, str):
                    norm_request["method"] = method
                else:
                    messages.append("context.request.method must be a string")
            for group in ("path", "query"):
                value = request.get(group)
                if value is not None:
                    if isinstance(value, dict):
                        norm_request[group] = value
                    else:
                        messages.append(
                            f"context.request.{group} must be an object mapping "
                            "names to values"
                        )
            headers = request.get("header")
            if headers is not None:
                if isinstance(headers, dict):
                    norm_request["header"] = headers
                else:
                    messages.append(
                        "context.request.header must be an object mapping "
                        "header names to values"
                    )
            if "body" in request:
                norm_request["body"] = request["body"]
            normalized["request"] = norm_request

    response = context.get("response")
    if response is not None:
        if not isinstance(response, dict):
            messages.append("context.response must be a JSON object")
        else:
            for field in sorted(response):
                if field not in _RESPONSE_FIELDS:
                    messages.append(
                        f"context.response.{field}: unknown response field; "
                        "expected one of statusCode/header/body"
                    )
            norm_response: Dict[str, Any] = {}
            status = response.get("statusCode")
            if status is not None:
                if isinstance(status, bool) or not isinstance(
                    status, (int, str)
                ):
                    messages.append(
                        "context.response.statusCode must be an integer or string"
                    )
                elif isinstance(status, str) and not status.strip():
                    messages.append(
                        "context.response.statusCode must not be an empty string"
                    )
                else:
                    norm_response["statusCode"] = status
            headers = response.get("header")
            if headers is not None:
                if isinstance(headers, dict):
                    norm_response["header"] = headers
                else:
                    messages.append(
                        "context.response.header must be an object mapping "
                        "header names to values"
                    )
            if "body" in response:
                norm_response["body"] = response["body"]
            normalized["response"] = norm_response

    return normalized, messages


# --------------------------------------------------------------------------- #
# Document shape / target resolution
# --------------------------------------------------------------------------- #


def _validate_shape(doc: Any) -> List[str]:
    if not isinstance(doc, dict):
        return ["document must be a JSON object"]
    messages: List[str] = []
    version = doc.get("openapi")
    if not isinstance(version, str) or not version.startswith("3."):
        messages.append(
            "document.openapi must be a version string starting with '3.' "
            f"(OpenAPI 3.x), got {version!r}"
        )
    if not isinstance(doc.get("paths"), dict):
        messages.append("document.paths must be an object")
    return messages


def _follow_chain(
    doc: Dict[str, Any],
    node: Any,
    node_pointer: str,
    is31: bool,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[str],
    Optional[Dict[str, str]],
    Optional[Dict[str, Any]],
]:
    """Follow a local ``$ref`` chain for a Response / Link Object.

    Returns ``(effective, terminal_pointer, provenance, error)`` where
    provenance maps each effective member to the JSON Pointer of the
    chain layer that provides it. OpenAPI 3.0 uses the terminal target
    verbatim (siblings of ``$ref`` ignored); OpenAPI 3.1 overlays
    sibling fields from the terminal back toward the occurrence, so
    nearer siblings win.
    """
    resolver = RefResolver(doc)
    layers: List[Tuple[Dict[str, Any], str]] = []
    current = node
    current_pointer = node_pointer
    seen: Dict[str, int] = {}
    refs: List[str] = []

    while isinstance(current, dict) and isinstance(current.get("$ref"), str):
        ref = current["$ref"]
        if not is_local_ref(ref):
            return None, None, None, {
                "code": "unresolvable_reference",
                "pointer": current_pointer,
                "ref": ref,
                "error": (
                    f"unsupported non-local reference '{ref}': only document-"
                    "local references of the form '#/...' are supported"
                ),
            }
        target_pointer = parse_local_ref(ref)
        if target_pointer in seen:
            loop_start = seen[target_pointer]
            loop = refs[loop_start:] + [ref]
            return None, None, None, {
                "code": "unresolvable_reference",
                "pointer": current_pointer,
                "ref": ref,
                "error": "circular reference detected: " + " -> ".join(loop),
            }
        if len(refs) >= MAX_REF_CHAIN:
            return None, None, None, {
                "code": "unresolvable_reference",
                "pointer": current_pointer,
                "ref": ref,
                "error": (
                    f"reference chain exceeds the maximum depth of "
                    f"{MAX_REF_CHAIN} links (last reference: {ref})"
                ),
            }
        seen[target_pointer] = len(refs)
        refs.append(ref)
        try:
            target = resolver.pointer(target_pointer)
        except UnresolvableReferenceError as exc:
            return None, None, None, {
                "code": "unresolvable_reference",
                "pointer": current_pointer,
                "ref": exc.ref,
                "error": str(exc),
            }
        if not isinstance(target, dict):
            return None, None, None, {
                "code": "unresolvable_reference",
                "pointer": current_pointer,
                "ref": ref,
                "error": f"reference '{ref}' does not target an object",
            }
        layers.append((current, current_pointer))
        current = target
        current_pointer = target_pointer

    if not isinstance(current, dict):
        return None, None, None, {
            "code": "invalid_node",
            "pointer": current_pointer,
            "error": "expected an object",
        }

    layers.append((current, current_pointer))
    if not is31:
        terminal_node, terminal_pointer = layers[-1]
        return (
            copy.deepcopy(terminal_node),
            terminal_pointer,
            dict.fromkeys(terminal_node, terminal_pointer),
            None,
        )

    effective: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}
    for layer_node, layer_pointer in reversed(layers):
        for key, value in layer_node.items():
            if key == "$ref":
                continue
            effective[key] = copy.deepcopy(value)
            provenance[key] = layer_pointer
    return effective, current_pointer, provenance, None


def _resolve_local_chain(
    doc: Dict[str, Any],
    node: Any,
    node_pointer: str,
    is31: bool,
    *,
    reject_root: bool = True,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[str],
    Optional[Dict[str, str]],
    Optional[Dict[str, Any]],
]:
    """Resolve a Response / Link style local ``$ref`` chain.

    Returns ``(effective, terminal_pointer, provenance, error)``. The
    effective object honors OpenAPI 3.1 sibling merging (3.0 uses the
    terminal verbatim); provenance locates each effective member.
    """
    effective, terminal_pointer, provenance, error = _follow_chain(
        doc, node, node_pointer, is31
    )
    if error is not None:
        return None, None, None, error
    if reject_root and terminal_pointer == "":
        return None, None, None, {
            "code": "unresolvable_reference",
            "pointer": node_pointer,
            "ref": "#",
            "error": (
                "reference '#' targets the document root; a Response or Link "
                "reference must point to a specific object location"
            ),
        }
    return effective, terminal_pointer, provenance, None


def _build_operation_id_index(
    doc: Dict[str, Any], is31: bool
) -> Tuple[
    Dict[str, Tuple[Dict[str, Any], str, str, str, str]],
    set,
    List[Dict[str, Any]],
]:
    """Map every operationId under ``paths`` to its operation.

    Returns ``(index, duplicates, path_item_errors)``; each index value is
    ``(operation, path, path_pointer, method_key, operation_pointer)``.
    A duplicated operationId is collected in *duplicates* and dropped
    from the index, making that name unusable (globally non-unique
    operationId). Path Item ``$ref`` chains are resolved cycle-safely;
    broken chains are gathered in *path_item_errors* and only surfaced
    when the targeted operationId cannot be found, so a Link is never
    blocked by an unrelated broken path item.
    """
    index: Dict[str, Tuple[Dict[str, Any], str, str, str, str]] = {}
    duplicates: set = set()
    path_item_errors: List[Dict[str, Any]] = []
    paths = doc.get("paths", {})

    for path in paths:
        path_pointer = pointer_join("/paths", path)
        raw_item = paths[path]
        if not isinstance(raw_item, dict):
            continue
        if isinstance(raw_item.get("$ref"), str):
            path_item, item_error = _resolve_path_item(doc, raw_item, path_pointer)
            if item_error is not None:
                path_item_errors.append(item_error)
                continue
        else:
            path_item = raw_item
        for key in path_item:
            if key.lower() not in HTTP_METHODS:
                continue
            operation = path_item[key]
            if not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId")
            if not isinstance(operation_id, str) or not operation_id:
                continue
            if operation_id in duplicates:
                continue
            if operation_id in index:
                duplicates.add(operation_id)
                index.pop(operation_id, None)
                continue
            index[operation_id] = (
                operation,
                path,
                path_pointer,
                key,
                pointer_join(path_pointer, key),
            )

    return index, duplicates, path_item_errors


def _resolve_operation_ref(
    doc: Dict[str, Any],
    operation_ref: Any,
    ref_pointer: str,
    is31: bool,
) -> Tuple[
    Optional[Tuple[Dict[str, Any], str, str, str, str]],
    Optional[Dict[str, Any]],
]:
    """Resolve a Link ``operationRef``.

    Returns ``(operation, path, path_pointer, method_key,
    operation_pointer)``.
    """
    if not isinstance(operation_ref, str) or not operation_ref:
        return None, {
            "code": "invalid_target",
            "pointer": ref_pointer,
            "error": "operationRef must be a non-empty string",
        }
    if not is_local_ref(operation_ref):
        return None, {
            "code": "unresolvable_reference",
            "pointer": ref_pointer,
            "ref": operation_ref,
            "error": (
                f"unsupported non-local operationRef '{operation_ref}': only "
                "document-local fragments of the form '#/paths/<path>/<method>' "
                "are supported"
            ),
        }
    pointer = parse_local_ref(operation_ref)
    segments = pointer.split("/")[1:]
    if (
        len(segments) != 3
        or segments[0] != "paths"
        or unescape_token(segments[2]).lower() not in HTTP_METHODS
    ):
        return None, {
            "code": "invalid_target",
            "pointer": ref_pointer,
            "ref": operation_ref,
            "error": (
                f"operationRef '{operation_ref}' must point at an operation, "
                "i.e. '#/paths/<url-path>/<method>'"
            ),
        }
    path = unescape_token(segments[1])
    method_key = unescape_token(segments[2])
    paths = doc.get("paths", {})
    if path not in paths:
        return None, {
            "code": "invalid_target",
            "pointer": ref_pointer,
            "ref": operation_ref,
            "error": f"operationRef '{operation_ref}' targets undeclared path '{path}'",
        }
    path_pointer = pointer_join("/paths", path)
    raw_item = paths[path]
    if isinstance(raw_item, dict) and isinstance(raw_item.get("$ref"), str):
        path_item, item_error = _resolve_path_item(doc, raw_item, path_pointer)
        if item_error is not None:
            error = dict(item_error)
            error["pointer"] = ref_pointer
            return None, error
    elif not isinstance(raw_item, dict):
        return None, {
            "code": "invalid_target",
            "pointer": ref_pointer,
            "ref": operation_ref,
            "error": f"operationRef '{operation_ref}' path item must be an object",
        }
    else:
        path_item = raw_item
    method_keys = {k.lower(): k for k in path_item if k.lower() in HTTP_METHODS}
    if method_key.lower() not in method_keys:
        return None, {
            "code": "invalid_target",
            "pointer": ref_pointer,
            "ref": operation_ref,
            "error": (
                f"operationRef '{operation_ref}': operation "
                f"{method_key.upper()} {path} is not declared"
            ),
        }
    actual_key = method_keys[method_key.lower()]
    operation = path_item[actual_key]
    if not isinstance(operation, dict):
        return None, {
            "code": "invalid_target",
            "pointer": ref_pointer,
            "ref": operation_ref,
            "error": f"operationRef '{operation_ref}' target must be an object",
        }
    operation_pointer = pointer_join(path_pointer, actual_key)
    return (
        operation,
        path,
        path_pointer,
        actual_key,
        operation_pointer,
    ), None


# --------------------------------------------------------------------------- #
# Parameter collection on the target operation
# --------------------------------------------------------------------------- #


def _collect_parameters(
    doc: Dict[str, Any],
    path_item: Dict[str, Any],
    operation: Dict[str, Any],
    path_pointer: str,
    op_pointer: str,
    is31: bool,
    errors: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Effective target parameters keyed by name (path-level then op-level).

    Operation-level parameters override path-level parameters of the same
    name regardless of location; names shared by parameters in different
    locations (or header names differing only in case) are flagged for
    the caller because Link mappings match by name alone.
    """
    occurrences: List[Tuple[Dict[str, Any], str, str]] = []

    def consume(raw_parameters: Any, base_pointer: str, level: str) -> None:
        if raw_parameters is None:
            return
        if not isinstance(raw_parameters, list):
            errors.append(
                {
                    "code": "invalid_parameter",
                    "pointer": base_pointer,
                    "error": "'parameters' must be an array",
                }
            )
            return
        for index, raw_param in enumerate(raw_parameters):
            origin_pointer = pointer_join(base_pointer, str(index))
            node, terminal_pointer, _provenance, error = _follow_chain(
                doc, raw_param, origin_pointer, is31
            )
            if error is not None:
                errors.append(error)
                continue
            if not isinstance(node, dict):
                errors.append(
                    {
                        "code": "invalid_parameter",
                        "pointer": origin_pointer,
                        "error": "parameter must be an object",
                    }
                )
                continue
            name = node.get("name")
            location = node.get("in")
            if not isinstance(name, str) or not name:
                errors.append(
                    {
                        "code": "invalid_parameter",
                        "pointer": terminal_pointer,
                        "error": "parameter is missing a string 'name'",
                    }
                )
                continue
            if location not in GROUPS:
                errors.append(
                    {
                        "code": "invalid_parameter",
                        "pointer": terminal_pointer,
                        "name": name,
                        "error": (
                            "parameter 'in' must be one of path/query/header/"
                            f"cookie, got {location!r}"
                        ),
                    }
                )
                continue
            occurrences.append((node, terminal_pointer, level))

    consume(
        path_item.get("parameters"),
        pointer_join(path_pointer, "parameters"),
        "path",
    )
    consume(
        operation.get("parameters"),
        pointer_join(op_pointer, "parameters"),
        "operation",
    )

    # Duplicate (in, name) within one parameters array is invalid, mirroring
    # /requests/materialize; an operation-level parameter legally overrides
    # the path-level parameter with the same (in, name).
    seen_exact: Dict[Tuple[str, str, str], str] = {}
    deduped: List[Tuple[Dict[str, Any], str, str]] = []
    for node, pointer, level in occurrences:
        exact = (level, node["in"], node["name"])
        if exact in seen_exact:
            errors.append(
                {
                    "code": "duplicate_parameter",
                    "pointer": pointer,
                    "first_pointer": seen_exact[exact],
                    "in": node["in"],
                    "name": node["name"],
                    "error": (
                        f"duplicate {level}-level {node['in']} parameter "
                        f"'{node['name']}'; a parameter list must not declare "
                        "the same (in, name) twice"
                    ),
                }
            )
            continue
        seen_exact[exact] = pointer
        deduped.append((node, pointer, level))

    # Operation-level overrides path-level on (in, name) (spec behavior),
    # while preserving document order of first introduction.
    op_level = {
        (n["in"], n["name"])
        for n, _p, level in deduped
        if level == "operation"
    }
    effective: List[Tuple[Dict[str, Any], str]] = []
    for node, _pointer, level in deduped:
        if level == "path" and (node["in"], node["name"]) in op_level:
            continue
        effective.append((node, _pointer))

    by_name: Dict[str, Dict[str, Any]] = {}
    for node, pointer in effective:
        by_name.setdefault(
            node["name"],
            {
                "name": node["name"],
                "node": node,
                "pointer": pointer,
                "matches": [],
            },
        )["matches"].append((node["in"], pointer))

    # Header parameters whose names differ only in case collide for
    # name-based Link matching as well.
    header_cases: Dict[str, List[str]] = {}
    for node, _pointer in effective:
        if node["in"] == "header":
            header_cases.setdefault(node["name"].lower(), []).append(node["name"])
    for lowered, names in header_cases.items():
        if len(set(names)) > 1:
            errors.append(
                {
                    "code": "duplicate_parameter",
                    "pointer": op_pointer,
                    "in": "header",
                    "name": sorted(names)[0],
                    "error": (
                        "header parameters "
                        + ", ".join(repr(n) for n in sorted(names))
                        + " differ only in case; Link parameter names would "
                        "be ambiguous"
                    ),
                }
            )
    return by_name


# --------------------------------------------------------------------------- #
# Response selection by status code
# --------------------------------------------------------------------------- #


def _canonical_status(raw: Any) -> Optional[str]:
    """Normalize the runtime status code to its 3-digit text form."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        text = str(raw)
    elif isinstance(raw, str) and raw.strip().isdigit():
        text = raw.strip()
    else:
        return None
    if len(text) != 3 or text[0] not in "12345":
        return None
    return text


def _select_response(
    responses: Dict[str, Any], status_text: str
) -> Optional[Tuple[str, str]]:
    """Pick the response key: exact code, then ``nXX`` range, then default.

    Returns ``(key, match_kind)`` with kind one of ``exact`` / ``range`` /
    ``default``. Range keys match case-insensitively (``2xx`` == ``2XX``).
    """
    if status_text in responses:
        return status_text, "exact"
    range_key = status_text[0] + "XX"
    for key in responses:
        if isinstance(key, str) and key.upper() == range_key:
            return key, "range"
    if "default" in responses:
        return "default", "default"
    return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def plan_link(
    doc: Any,
    selector: Any,
    link_name: Any,
    context: Any = None,
) -> Dict[str, Any]:
    """Build the follow-up call plan for one Link of the source operation.

    :raises InvalidDocumentError: *doc* is not OpenAPI 3.x.
    :raises InvalidSelectorError: the source selector is malformed/unknown.
    :raises InvalidLinkRequestError: *link_name* / *context* malformed.
    :raises LinkPlanningError: semantic problems (422), each carrying
        source JSON Pointers.
    """
    shape_errors = _validate_shape(doc)
    if shape_errors:
        raise InvalidDocumentError(shape_errors)

    parsed = parse_selector(selector)
    if parsed is None:
        raise InvalidSelectorError(
            [
                f"malformed selector {selector!r}: expected 'METHOD /path' "
                "(e.g. 'GET /pets/{petId}')"
            ]
        )
    method, path = parsed
    if path not in doc["paths"]:
        raise InvalidSelectorError(
            [f"{method.upper()} {path}: path is not declared"]
        )

    request_errors: List[str] = []
    if not isinstance(link_name, str) or not link_name:
        request_errors.append(
            "link_name must be the non-empty name of a Link in the source "
            "operation's response"
        )
    if context is None:
        context = {}
    normalized_context, context_errors = normalize_context(context)
    request_errors.extend(context_errors)
    if request_errors:
        raise InvalidLinkRequestError(request_errors)

    is31 = str(doc.get("openapi", "")).startswith(("3.1", "3.2"))
    errors: List[Dict[str, Any]] = []

    # --- Locate the source operation (path item $ref cycle-safe) ------- #
    path_pointer = pointer_join("/paths", path)
    source_path_item, item_error = _resolve_path_item(
        doc, doc["paths"][path], path_pointer
    )
    if item_error is not None:
        raise LinkPlanningError(_sorted_errors([item_error]))

    method_keys = {
        key.lower(): key for key in source_path_item if key.lower() in HTTP_METHODS
    }
    if method not in method_keys:
        raise InvalidSelectorError(
            [f"{method.upper()} {path}: operation is not declared"]
        )
    operation_key = method_keys[method]
    source_operation = source_path_item[operation_key]
    operation_pointer = pointer_join(path_pointer, operation_key)
    if not isinstance(source_operation, dict):
        raise LinkPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "invalid_node",
                        "pointer": operation_pointer,
                        "error": "operation must be an object",
                    }
                ]
            )
        )
    source_operation_pointer = operation_pointer

    responses = source_operation.get("responses")
    if not isinstance(responses, dict):
        raise LinkPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "invalid_response",
                        "pointer": pointer_join(operation_pointer, "responses"),
                        "error": "source operation must declare a 'responses' object",
                    }
                ]
            )
        )

    # --- Choose the response by runtime status code -------------------- #
    response_msg = normalized_context.get("response")
    if not isinstance(response_msg, dict) or "statusCode" not in response_msg:
        raise LinkPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "missing_status_code",
                        "pointer": "/context/response/statusCode",
                        "error": (
                            "a response status code is required to select the "
                            "source response (exact, nXX range or default)"
                        ),
                    }
                ]
            )
        )
    status_text = _canonical_status(response_msg["statusCode"])
    if status_text is None:
        raise LinkPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "invalid_status",
                        "pointer": "/context/response/statusCode",
                        "error": (
                            f"status code {response_msg['statusCode']!r} must be "
                            "a 3-digit HTTP status code (1xx..5xx)"
                        ),
                    }
                ]
            )
        )

    selected = _select_response(responses, status_text)
    if selected is None:
        raise LinkPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "response_not_declared",
                        "pointer": pointer_join(operation_pointer, "responses"),
                        "status_code": status_text,
                        "error": (
                            f"source operation declares no response for status "
                            f"code {status_text} (no exact, "
                            f"{status_text[0]}XX range or default response)"
                        ),
                    }
                ]
            )
        )
    response_key, match_kind = selected
    response_ref_pointer = pointer_join(
        pointer_join(operation_pointer, "responses"), response_key
    )
    (
        response_obj,
        response_terminal,
        response_provenance,
        response_error,
    ) = _resolve_local_chain(
        doc, responses[response_key], response_ref_pointer, is31
    )
    if response_error is not None:
        raise LinkPlanningError(_sorted_errors([response_error]))
    if not isinstance(response_obj, dict):
        raise LinkPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "invalid_response",
                        "pointer": response_terminal or response_ref_pointer,
                        "error": "response must be an object",
                    }
                ]
            )
        )

    # --- Resolve the named Link Object --------------------------------- #
    links = response_obj.get("links")
    links_origin = (
        response_provenance.get("links")
        if response_provenance is not None
        else response_terminal
    )
    links_base_pointer = pointer_join(
        links_origin or response_ref_pointer, "links"
    )
    if not isinstance(links, dict) or link_name not in links:
        raise LinkPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "unknown_link",
                        "pointer": links_base_pointer,
                        "name": link_name,
                        "error": (
                            f"response for status {status_text} declares no link "
                            f"named {link_name!r}"
                        ),
                    }
                ]
            )
        )
    link_ref_pointer = pointer_join(links_base_pointer, link_name)
    (
        link,
        link_terminal,
        link_provenance,
        link_error,
    ) = _resolve_local_chain(
        doc, links[link_name], link_ref_pointer, is31
    )
    if link_error is not None:
        raise LinkPlanningError(_sorted_errors([link_error]))
    if not isinstance(link, dict):
        raise LinkPlanningError(
            _sorted_errors(
                [
                    {
                    "code": "invalid_link",
                    "pointer": link_terminal or link_ref_pointer,
                    "error": "link must be a Link Object",
                    }
                ]
            )
        )

    def _member_pointer(member: str) -> str:
        """JSON Pointer where an effective link member is declared."""
        origin = (
            link_provenance.get(member)
            if link_provenance is not None
            else link_terminal
        )
        return pointer_join(origin or link_ref_pointer, member)

    # --- Resolve the target operation: operationRef XOR operationId ---- #
    has_ref = "operationRef" in link
    has_id = "operationId" in link
    if has_ref == has_id:
        errors.append(
            {
                "code": "invalid_link",
                "pointer": link_terminal or link_ref_pointer,
                "name": link_name,
                "error": (
                    "a Link Object must contain exactly one of 'operationRef' "
                    "or 'operationId'"
                ),
            }
        )
        target: Optional[Tuple[Dict[str, Any], str, str, str, str]] = None
    elif has_ref:
        op_ref_pointer = _member_pointer("operationRef")
        target, target_error = _resolve_operation_ref(
            doc, link.get("operationRef"), op_ref_pointer, is31
        )
        if target_error is not None:
            errors.append(target_error)
    else:
        operation_id = link.get("operationId")
        op_id_pointer = _member_pointer("operationId")
        id_index, id_duplicates, id_path_errors = _build_operation_id_index(
            doc, is31
        )
        if not isinstance(operation_id, str) or not operation_id:
            errors.append(
                {
                    "code": "invalid_link",
                    "pointer": op_id_pointer,
                    "name": link_name,
                    "error": "'operationId' must be a non-empty string",
                }
            )
            target = None
        elif operation_id in id_duplicates:
            errors.append(
                {
                    "code": "ambiguous_operation",
                    "pointer": op_id_pointer,
                    "ref": operation_id,
                    "error": (
                        f"operationId '{operation_id}' is declared by more than "
                        "one operation; operationId values must be globally unique"
                    ),
                }
            )
            target = None
        elif operation_id not in id_index:
            errors.append(
                {
                    "code": "invalid_target",
                    "pointer": op_id_pointer,
                    "ref": operation_id,
                    "error": (
                        f"operationId '{operation_id}' is not declared by any "
                        "operation under 'paths'"
                    ),
                }
            )
            # A broken Path Item chain (e.g. circular component ref) may be
            # the reason the id could not be indexed; surface it as well.
            errors.extend(id_path_errors)
            target = None
        else:
            target = id_index[operation_id]

    if target is None:
        raise LinkPlanningError(_sorted_errors(errors))

    (
        target_operation,
        target_path,
        target_path_pointer,
        target_method_key,
        target_op_pointer,
    ) = target
    if not isinstance(target_operation, dict):
        raise LinkPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "invalid_target",
                        "pointer": target_op_pointer,
                        "error": "target operation must be an object",
                    }
                ]
            )
        )

    # Re-resolve the target path item fresh so its effective parameters are
    # merged exactly as the source-side machinery would see them.
    target_item_error: Optional[Dict[str, Any]] = None
    raw_target_item = doc["paths"].get(target_path)
    if isinstance(raw_target_item, dict) and isinstance(
        raw_target_item.get("$ref"), str
    ):
        target_path_item, target_item_error = _resolve_path_item(
            doc, raw_target_item, target_path_pointer
        )
    elif isinstance(raw_target_item, dict):
        target_path_item = raw_target_item
    else:
        target_path_item, target_item_error = None, {
            "code": "invalid_target",
            "pointer": target_path_pointer,
            "error": "target path item must be an object",
        }
    if target_item_error is not None:
        raise LinkPlanningError(_sorted_errors([target_item_error]))
    if target_method_key not in target_path_item:
        raise LinkPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "invalid_target",
                        "pointer": target_op_pointer,
                        "error": "target operation is not declared on the path item",
                    }
                ]
            )
        )
    target_operation = target_path_item[target_method_key]
    if not isinstance(target_operation, dict):
        raise LinkPlanningError(
            _sorted_errors(
                [
                    {
                        "code": "invalid_target",
                        "pointer": target_op_pointer,
                        "error": "target operation must be an object",
                    }
                ]
            )
        )

    by_name = _collect_parameters(
        doc,
        target_path_item,
        target_operation,
        target_path_pointer,
        target_op_pointer,
        is31,
        errors,
    )

    # --- Evaluate Link.parameters -------------------------------------- #
    raw_parameters = link.get("parameters", {})
    parameters_pointer = _member_pointer("parameters")
    if not isinstance(raw_parameters, dict):
        errors.append(
            {
                "code": "invalid_link",
                "pointer": parameters_pointer,
                "error": "link 'parameters' must be an object",
            }
        )
        raw_parameters = {}

    grouped: Dict[str, Dict[str, Any]] = {group: {} for group in GROUPS}
    mappings: List[Dict[str, Any]] = []

    for param_key in raw_parameters:
        mapping_pointer = pointer_join(parameters_pointer, param_key)
        raw_expression = raw_parameters[param_key]
        if not isinstance(param_key, str) or not param_key:
            errors.append(
                {
                    "code": "invalid_link",
                    "pointer": mapping_pointer,
                    "error": "link parameter key must be a non-empty string",
                }
            )
            continue

        # Exact-name match first; header names then match case-insensitively.
        entry = by_name.get(param_key)
        if entry is None:
            header_match = _match_header_name(by_name, param_key)
            if header_match is not None:
                entry = header_match
        if entry is None:
            errors.append(
                {
                    "code": "unknown_parameter",
                    "pointer": mapping_pointer,
                    "name": param_key,
                    "error": (
                        f"target operation {target_method_key.upper()} "
                        f"{target_path} declares no parameter named "
                        f"{param_key!r}"
                    ),
                }
            )
            descriptor = parse_link_expression(raw_expression)
            if descriptor is None:
                errors.append(
                    {
                        "code": "invalid_expression",
                        "pointer": mapping_pointer,
                        "name": param_key,
                        "expression": _expression_text(raw_expression),
                        "error": (
                            f"link parameter {param_key!r} value must be a "
                            "single runtime expression (e.g. "
                            "'$request.body#/id')"
                        ),
                    }
                )
            continue

        matches = entry["matches"]
        location = _single_location(matches)
        if location is None:
            errors.append(
                {
                    "code": "ambiguous_parameter",
                    "pointer": mapping_pointer,
                    "name": param_key,
                    "error": _ambiguity_message(param_key, matches),
                }
            )
            continue

        descriptor = parse_link_expression(raw_expression)
        if descriptor is None:
            errors.append(
                {
                    "code": "invalid_expression",
                    "pointer": mapping_pointer,
                    "name": param_key,
                    "expression": _expression_text(raw_expression),
                    "error": (
                        f"link parameter {param_key!r} value must be a single "
                        "runtime expression (e.g. '$request.body#/id'); URL "
                        "templates and literal strings are not allowed"
                    ),
                }
            )
            continue

        status, value, context_pointer = evaluate_expression(
            descriptor, normalized_context
        )
        if status == "missing":
            errors.append(
                {
                    "code": "missing_value",
                    "pointer": mapping_pointer,
                    "name": param_key,
                    "expression": raw_expression,
                    "context_pointer": context_pointer,
                    "error": (
                        f"runtime expression {raw_expression!r} for parameter "
                        f"{param_key!r} has no value in the supplied "
                        "request/response context"
                    ),
                }
            )
            continue
        if status == "ambiguous":
            errors.append(
                {
                    "code": "ambiguous_header",
                    "pointer": mapping_pointer,
                    "name": param_key,
                    "expression": raw_expression,
                    "context_pointer": context_pointer,
                    "error": (
                        "runtime expression header name matches several "
                        "context headers that differ only in case"
                    ),
                }
            )
            continue

        grouped[location][entry["name"]] = copy.deepcopy(value)
        mappings.append(
            {
                "in": location,
                "name": entry["name"],
                "mapping_key": param_key,
                "pointer": mapping_pointer,
                "expression": raw_expression,
                "parameter_pointer": entry["pointer"],
                "context_pointer": context_pointer,
            }
        )

    # --- Required parameters without a mapping are an error ----------- #
    for name, entry in by_name.items():
        matches = entry["matches"]
        location = _single_location(matches)
        if location is None:
            continue  # ambiguous_parameter already reported above
        node = entry["node"]
        required = location == "path" or bool(node.get("required"))
        if required and name not in grouped[location]:
            errors.append(
                {
                    "code": "missing_required_value",
                    "pointer": entry["pointer"],
                    "in": location,
                    "name": name,
                    "error": (
                        f"required {location} parameter '{name}' of the target "
                        "operation is not assigned by the link"
                    ),
                }
            )

    # --- Evaluate Link.requestBody ------------------------------------- #
    request_body = None
    request_body_mapping: Optional[Dict[str, Any]] = None
    if "requestBody" in link:
        body_pointer = _member_pointer("requestBody")
        raw_body_expression = link["requestBody"]
        descriptor = parse_link_expression(raw_body_expression)
        if descriptor is None:
            errors.append(
                {
                    "code": "invalid_expression",
                    "pointer": body_pointer,
                    "expression": _expression_text(raw_body_expression),
                    "error": (
                        "link 'requestBody' must be a single runtime "
                        "expression (e.g. '$request.body')"
                    ),
                }
            )
        else:
            status, value, context_pointer = evaluate_expression(
                descriptor, normalized_context
            )
            if status == "missing":
                errors.append(
                    {
                        "code": "missing_value",
                        "pointer": body_pointer,
                        "expression": raw_body_expression,
                        "context_pointer": context_pointer,
                        "error": (
                            f"runtime expression {raw_body_expression!r} for the "
                            "follow-up request body has no value in the supplied "
                            "request/response context"
                        ),
                    }
                )
            elif status == "ambiguous":
                errors.append(
                    {
                        "code": "ambiguous_header",
                        "pointer": body_pointer,
                        "expression": raw_body_expression,
                        "context_pointer": context_pointer,
                        "error": (
                            "runtime expression header name matches several "
                            "context headers that differ only in case"
                        ),
                    }
                )
            else:
                request_body = copy.deepcopy(value)
                request_body_mapping = {
                    "pointer": body_pointer,
                    "expression": raw_body_expression,
                    "context_pointer": context_pointer,
                }

    # --- Target requestBody $ref sanity (cycle-safe) ------------------- #
    target_body = target_operation.get("requestBody")
    if target_body is not None:
        target_body_pointer = pointer_join(target_op_pointer, "requestBody")
        if isinstance(target_body, dict) and isinstance(
            target_body.get("$ref"), str
        ):
            _resolved, _term, _prov, body_error = _resolve_local_chain(
                doc, target_body, target_body_pointer, is31
            )
            if body_error is not None:
                errors.append(body_error)

    if errors:
        raise LinkPlanningError(_sorted_errors(errors))

    mappings.sort(
        key=lambda item: (item["in"], item["name"], item["mapping_key"])
    )

    return {
        "link_name": link_name,
        "link_pointer": link_ref_pointer,
        "response": {
            "status_code": status_text,
            "key": response_key,
            "match": match_kind,
            "pointer": response_ref_pointer,
        },
        "target": {
            "method": target_method_key.lower(),
            "path": target_path,
            "operation_pointer": target_op_pointer,
            "operation_id": target_operation.get("operationId"),
        },
        "parameters": {group: grouped[group] for group in GROUPS},
        "request_body": request_body,
        "mappings": mappings,
        "request_body_mapping": request_body_mapping,
    }


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _expression_text(raw: Any) -> Any:
    return raw if isinstance(raw, str) else None


def _single_location(
    matches: List[Tuple[str, str]]
) -> Optional[str]:
    """The unique parameter location for a name, else ``None`` (ambiguous)."""
    locations = {location for location, _pointer in matches}
    if len(locations) == 1:
        return next(iter(locations))
    return None


def _ambiguity_message(name: str, matches: List[Tuple[str, str]]) -> str:
    rendered = ", ".join(
        f"{location} @ {pointer}" for location, pointer in sorted(matches)
    )
    return (
        f"parameter name '{name}' is declared in multiple locations ({rendered}); "
        "link parameters match by name alone, so the target is ambiguous"
    )


def _match_header_name(
    by_name: Dict[str, Dict[str, Any]], name: str
) -> Optional[Dict[str, Any]]:
    """Case-insensitive match against header parameter names only."""
    lowered = name.lower()
    hits = [
        key
        for key, entry in by_name.items()
        if _single_location(entry["matches"]) == "header"
        and key.lower() == lowered
    ]
    if len(hits) == 1:
        return by_name[hits[0]]
    return None


def _sorted_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(item: Dict[str, Any]):
        return (
            str(item.get("code")),
            str(item.get("in") or ""),
            str(item.get("name") or ""),
            str(item.get("pointer") or ""),
            str(item.get("expression") or ""),
            str(item.get("ref") or ""),
            str(item.get("error") or ""),
        )

    return sorted(errors, key=key)
