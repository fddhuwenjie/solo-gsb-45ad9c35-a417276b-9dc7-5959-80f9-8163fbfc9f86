"""Regression tests for ``POST /links/plan``.

Covers response selection by status code (exact / nXX range / default),
Link / Response / Path Item local ``$ref`` resolution (3.0 vs 3.1
siblings, dangling / external / circular), operationRef vs globally
unique operationId targeting, runtime-expression evaluation in Link
``parameters`` / ``requestBody`` (all sources, JSON Pointer fragments,
case-insensitive headers, structured values), name-based parameter
matching incl. cross-location ambiguity, required parameter checks,
400/422 boundaries, stable mapping provenance and the non-mutation
guarantee.
"""

from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def target_get(
    path="/orders/{orderId}",
    operation_id="getOrder",
    method="get",
    parameters=None,
    extra=None,
):
    operation = {
        "operationId": operation_id,
        "parameters": parameters
        if parameters is not None
        else [
            {
                "name": "orderId",
                "in": "path",
                "required": True,
                "schema": {"type": "string"},
            }
        ],
        "responses": {"200": {"description": "ok"}},
    }
    if extra:
        operation.update(extra)
    return {path: {method: operation}}


def make_doc(
    link=None,
    *,
    responses=None,
    openapi="3.0.3",
    targets=None,
    components=None,
    source_path="/orders",
    source_method="post",
    source_operation_id="createOrder",
    source_extra=None,
    source_item_extra=None,
    target_extra_paths=None,
):
    if link is None:
        link = {
            "operationId": "getOrder",
            "parameters": {"orderId": "$response.body#/id"},
        }
    if responses is None:
        responses = {
            "201": {"description": "created", "links": {"getOrder": link}},
            "2XX": {"description": "accepted"},
            "default": {"description": "error"},
        }
    source_operation = {
        "operationId": source_operation_id,
        "responses": responses,
    }
    if source_extra:
        source_operation.update(source_extra)
    source_item = {source_method: source_operation}
    if source_item_extra:
        source_item.update(source_item_extra)
    doc = {
        "openapi": openapi,
        "info": {"title": "Orders", "version": "1.0"},
        "paths": {source_path: source_item},
    }
    doc["paths"].update(targets if targets is not None else target_get())
    if target_extra_paths:
        doc["paths"].update(target_extra_paths)
    if components is not None:
        doc["components"] = components
    return doc


def ctx_obj(body=None, *, status_code=201, **overrides):
    ctx = {"response": {"statusCode": status_code}}
    if body is not None:
        ctx["response"]["body"] = body
    ctx.update(overrides)
    return ctx


def plan(
    doc,
    ctx=None,
    *,
    selector="POST /orders",
    link_name="getOrder",
    context=None,
):
    payload = {
        "document": doc,
        "selector": selector,
        "link_name": link_name,
    }
    ctx = ctx if ctx is not None else context
    if isinstance(ctx, dict) and "request" not in ctx and "response" not in ctx:
        # Shorthand: a bare dict is the runtime response body.
        ctx = {"response": {"statusCode": 201, "body": ctx}}
    if ctx is not None:
        payload["context"] = ctx
    return client.post("/links/plan", json=payload)


def errors_of(response):
    return {(e["code"], e.get("name")) for e in response.json()["errors"]}


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_basic_operation_id_plan():
    doc = make_doc()
    response = plan(doc, context={"id": "ord-42"})
    assert response.status_code == 200
    body = response.json()
    assert body["link_name"] == "getOrder"
    assert body["link_pointer"] == (
        "/paths/~1orders/post/responses/201/links/getOrder"
    )
    assert body["response"] == {
        "status_code": "201",
        "key": "201",
        "match": "exact",
        "pointer": "/paths/~1orders/post/responses/201",
    }
    assert body["target"] == {
        "method": "get",
        "path": "/orders/{orderId}",
        "operation_pointer": "/paths/~1orders~1{orderId}/get",
        "operation_id": "getOrder",
    }
    assert body["parameters"]["path"] == {"orderId": "ord-42"}
    assert body["parameters"]["query"] == {}
    assert body["parameters"]["header"] == {}
    assert body["parameters"]["cookie"] == {}
    assert body["request_body"] is None
    assert body["request_body_mapping"] is None
    (mapping,) = body["mappings"]
    assert mapping == {
        "in": "path",
        "name": "orderId",
        "mapping_key": "orderId",
        "pointer": (
            "/paths/~1orders/post/responses/201/links/getOrder/parameters/orderId"
        ),
        "expression": "$response.body#/id",
        "parameter_pointer": (
            "/paths/~1orders~1{orderId}/get/parameters/0"
        ),
        "context_pointer": "/context/response/body/id",
    }


def test_response_selection_range_and_default():
    # Only 2XX + default declared: 201 matches the range.
    doc = make_doc(
        responses={
            "2XX": {
                "description": "accepted",
                "links": {
                    "getOrder": {
                        "operationId": "getOrder",
                        "parameters": {"orderId": "$response.body#/id"},
                    }
                },
            },
            "default": {"description": "error"},
        }
    )
    response = plan(doc, context={"id": "x"})
    assert response.status_code == 200
    assert response.json()["response"] == {
        "status_code": "201",
        "key": "2XX",
        "match": "range",
        "pointer": "/paths/~1orders/post/responses/2XX",
    }

    # Range codes are case-insensitive (declared lowercase).
    doc = make_doc(
        responses={
            "2xx": {
                "description": "accepted",
                "links": {
                    "getOrder": {
                        "operationId": "getOrder",
                        "parameters": {"orderId": "$response.body#/id"},
                    }
                },
            }
        }
    )
    response = plan(doc, ctx_obj({"id": "x"}, status_code=200))
    assert response.status_code == 200
    assert response.json()["response"]["key"] == "2xx"
    assert response.json()["response"]["match"] == "range"

    # Exact code wins over a range and default.
    doc = make_doc()
    response = plan(doc, ctx_obj({"id": "x"}, status_code=201))
    assert response.json()["response"]["match"] == "exact"

    # 404 falls through to default.
    doc = make_doc(
        responses={
            "201": {"description": "created"},
            "default": {
                "description": "error",
                "links": {
                    "getOrder": {
                        "operationId": "getOrder",
                        "parameters": {"orderId": "$response.body#/id"},
                    }
                },
            },
        }
    )
    response = plan(doc, ctx_obj({"id": "d"}, status_code=404))
    assert response.status_code == 200
    assert response.json()["response"] == {
        "status_code": "404",
        "key": "default",
        "match": "default",
        "pointer": "/paths/~1orders/post/responses/default",
    }


def test_status_code_accepted_as_string():
    doc = make_doc()
    response = plan(doc, ctx_obj({"id": "x"}, status_code="201"))
    assert response.status_code == 200
    assert response.json()["response"]["status_code"] == "201"


def test_operation_ref_target():
    link = {
        "operationRef": "#/paths/~1orders~1{orderId}/get",
        "parameters": {"orderId": "$response.body#/id"},
    }
    doc = make_doc(link)
    response = plan(doc, context={"id": "ord-7"})
    assert response.status_code == 200
    body = response.json()
    assert body["target"]["operation_pointer"] == (
        "/paths/~1orders~1{orderId}/get"
    )
    assert body["parameters"]["path"] == {"orderId": "ord-7"}


def test_all_expression_sources():
    targets = target_get(
        parameters=[
            {"name": "orderId", "in": "path", "required": True},
            {"name": "q", "in": "query"},
            {"name": "X-Trace", "in": "header"},
            {"name": "session", "in": "cookie"},
        ]
    )
    link = {
        "operationId": "getOrder",
        "parameters": {
            "orderId": "$response.body#/id",
            "q": "$request.query.q",
            "x-trace": "$request.header.x-trace",
            "session": "$response.header.Set-Cookie",
        },
    }
    doc = make_doc(link, targets=targets)
    ctx = {
        "request": {
            "query": {"q": "five"},
            "header": {"X-Trace": "tr"},
        },
        "response": {
            "statusCode": 201,
            "header": {"Set-Cookie": "sess=1"},
            "body": {"id": "o"},
        },
    }
    response = plan(doc, context=ctx)
    assert response.status_code == 200
    grouped = response.json()["parameters"]
    assert grouped["path"] == {"orderId": "o"}
    assert grouped["query"] == {"q": "five"}
    assert grouped["header"] == {"X-Trace": "tr"}
    assert grouped["cookie"] == {"session": "sess=1"}


def test_request_path_source_and_url_method_status_expressions():
    targets = target_get(
        parameters=[
            {"name": "orderId", "in": "path", "required": True},
            {"name": "via", "in": "query"},
        ]
    )
    link = {
        "operationId": "getOrder",
        "parameters": {
            "orderId": "$request.path.oid",
            "via": "$method",
        },
    }
    doc = make_doc(link, targets=targets)
    ctx = {
        "request": {"method": "POST", "path": {"oid": "p42"}},
        "response": {"statusCode": 201},
    }
    response = plan(doc, context=ctx)
    assert response.status_code == 200
    grouped = response.json()["parameters"]
    assert grouped["path"] == {"orderId": "p42"}
    assert grouped["query"] == {"via": "POST"}


def test_header_mapping_key_case_insensitive_uses_declared_name():
    targets = target_get(
        parameters=[
            {"name": "orderId", "in": "path", "required": True},
            {"name": "X-Trace", "in": "header"},
        ]
    )
    link = {
        "operationId": "getOrder",
        "parameters": {
            "orderId": "$response.body#/id",
            "x-TRACE": "$request.header.x-trace",
        },
    }
    doc = make_doc(link, targets=targets)
    ctx = {
        "request": {"header": {"X-Trace": "t"}},
        "response": {"statusCode": 201, "body": {"id": "x"}},
    }
    response = plan(doc, context=ctx)
    assert response.status_code == 200
    assert response.json()["parameters"]["header"] == {"X-Trace": "t"}
    header_mapping = [
        m for m in response.json()["mappings"] if m["in"] == "header"
    ][0]
    assert header_mapping["name"] == "X-Trace"
    assert header_mapping["mapping_key"] == "x-TRACE"


def test_structured_expression_values_pass_through():
    targets = target_get(
        parameters=[
            {"name": "orderId", "in": "path", "required": True},
            {"name": "tags", "in": "query"},
        ]
    )
    link = {
        "operationId": "getOrder",
        "parameters": {
            "orderId": "$response.body#/id",
            "tags": "$response.body#/tags",
        },
    }
    doc = make_doc(link, targets=targets)
    response = plan(
        doc,
        context={"id": "x", "tags": [1, 2, {"deep": True}]},
    )
    assert response.status_code == 200
    assert response.json()["parameters"]["query"] == {
        "tags": [1, 2, {"deep": True}]
    }


def test_explicit_json_null_is_a_value():
    targets = target_get(
        parameters=[{"name": "orderId", "in": "query"}]
    )
    link = {
        "operationId": "getOrder",
        "parameters": {"orderId": "$response.body#/id"},
    }
    doc = make_doc(link, targets=targets)
    response = plan(doc, context={"id": None})
    assert response.status_code == 200
    assert response.json()["parameters"]["query"] == {"orderId": None}


def test_request_body_expression():
    targets = target_get(
        method="put",
        parameters=[{"name": "orderId", "in": "path", "required": True}],
        extra={
            "operationId": "replaceOrder",
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
        },
    )
    link = {
        "operationId": "replaceOrder",
        "parameters": {"orderId": "$response.body#/id"},
        "requestBody": "$request.body",
    }
    doc = make_doc(link, targets=targets)
    ctx = {
        "request": {"body": {"item": "v", "count": 2}},
        "response": {"statusCode": 201, "body": {"id": "o"}},
    }
    response = plan(doc, context=ctx)
    assert response.status_code == 200
    body = response.json()
    assert body["request_body"] == {"item": "v", "count": 2}
    assert body["request_body_mapping"] == {
        "pointer": (
            "/paths/~1orders/post/responses/201/links/getOrder/requestBody"
        ),
        "expression": "$request.body",
        "context_pointer": "/context/request/body",
    }


def test_request_body_json_pointer_fragment():
    targets = target_get(
        method="put",
        parameters=[{"name": "orderId", "in": "path", "required": True}],
        extra={"operationId": "replaceOrder"},
    )
    link = {
        "operationId": "replaceOrder",
        "parameters": {"orderId": "$response.body#/id"},
        "requestBody": "$response.body#/payload",
    }
    doc = make_doc(link, targets=targets)
    response = plan(
        doc, context={"id": "o", "payload": {"hello": "world"}}
    )
    assert response.status_code == 200
    assert response.json()["request_body"] == {"hello": "world"}
    assert (
        response.json()["request_body_mapping"]["context_pointer"]
        == "/context/response/body/payload"
    )


def test_path_level_parameters_merge_into_target():
    # The path parameter is declared at path level only.
    targets = {
        "/orders/{orderId}": {
            "parameters": [
                {"name": "orderId", "in": "path", "required": True}
            ],
            "get": {
                "operationId": "getOrder",
                "responses": {"200": {"description": "ok"}},
            },
        }
    }
    doc = make_doc(targets=targets)
    response = plan(doc, context={"id": "z"})
    assert response.status_code == 200
    assert response.json()["parameters"]["path"] == {"orderId": "z"}
    assert (
        response.json()["mappings"][0]["parameter_pointer"]
        == "/paths/~1orders~1{orderId}/parameters/0"
    )


def test_parameter_component_refs_followed():
    components = {
        "parameters": {
            "OrderId": {
                "name": "orderId",
                "in": "path",
                "required": True,
                "schema": {"type": "string"},
            }
        }
    }
    targets = target_get(
        parameters=[{"$ref": "#/components/parameters/OrderId"}]
    )
    doc = make_doc(targets=targets, components=components)
    response = plan(doc, context={"id": "c9"})
    assert response.status_code == 200
    assert response.json()["parameters"]["path"] == {"orderId": "c9"}
    assert (
        response.json()["mappings"][0]["parameter_pointer"]
        == "/components/parameters/OrderId"
    )


def test_response_and_link_component_refs_3_0():
    components = {
        "links": {
            "GetOrder": {
                "operationId": "getOrder",
                "parameters": {"orderId": "$response.body#/id"},
            }
        },
        "responses": {
            "Created": {
                "description": "created",
                "links": {"getOrder": {"$ref": "#/components/links/GetOrder"}},
            }
        },
    }
    doc = make_doc(
        responses={"201": {"$ref": "#/components/responses/Created"}},
        components=components,
    )
    response = plan(doc, context={"id": "ref-1"})
    assert response.status_code == 200
    body = response.json()
    # The link entry pointer stays at the $ref occurrence; mapping
    # pointers follow to the component definition under OpenAPI 3.0.
    assert body["link_pointer"] == (
        "/components/responses/Created/links/getOrder"
    )
    assert body["mappings"][0]["pointer"] == (
        "/components/links/GetOrder/parameters/orderId"
    )
    assert body["parameters"]["path"] == {"orderId": "ref-1"}


def test_openapi_3_1_link_sibling_fields_override():
    components = {
        "links": {
            "GetOrder": {
                "operationId": "getOrder",
                "parameters": {"orderId": "$response.body#/oldId"},
            }
        }
    }
    doc = make_doc(
        openapi="3.1.0",
        responses={
            "201": {
                "description": "created",
                "links": {
                    "getOrder": {
                        "$ref": "#/components/links/GetOrder",
                        "parameters": {"orderId": "$response.body#/id"},
                    }
                },
            }
        },
        components=components,
    )
    response = plan(doc, context={"id": "new", "oldId": "old"})
    assert response.status_code == 200
    body = response.json()
    assert body["parameters"]["path"] == {"orderId": "new"}
    # The mapping provenance points at the nearer sibling layer.
    assert body["mappings"][0]["pointer"] == (
        "/paths/~1orders/post/responses/201/links/getOrder/parameters/orderId"
    )
    assert body["mappings"][0]["expression"] == "$response.body#/id"


def test_3_1_path_item_ref_provides_target():
    targets = target_get()
    doc = make_doc(
        openapi="3.1.0",
        targets=targets,
        components={"pathItems": {"OrderItem": targets["/orders/{orderId}"]}},
    )
    doc["paths"]["/orders/{orderId}"] = {
        "$ref": "#/components/pathItems/OrderItem"
    }
    response = plan(doc, context={"id": "p"})
    assert response.status_code == 200
    assert (
        response.json()["target"]["operation_pointer"]
        == "/paths/~1orders~1{orderId}/get"
    )


def test_3_1_path_item_ref_sibling_parameters():
    # Path-level parameters live on the referenced path item component.
    targets = {
        "/orders/{orderId}": {
            "parameters": [
                {"name": "orderId", "in": "path", "required": True}
            ],
            "get": {
                "operationId": "getOrder",
                "responses": {"200": {"description": "ok"}},
            },
        }
    }
    doc = make_doc(
        openapi="3.1.0",
        targets=targets,
        components={"pathItems": {"OrderItem": targets["/orders/{orderId}"]}},
    )
    doc["paths"]["/orders/{orderId}"] = {
        "$ref": "#/components/pathItems/OrderItem"
    }
    response = plan(doc, context={"id": "p"})
    assert response.status_code == 200
    assert response.json()["parameters"]["path"] == {"orderId": "p"}


def test_pct_encoded_query_and_path_names():
    targets = target_get(
        parameters=[{"name": "orderId", "in": "path", "required": True}]
    )
    link = {
        "operationId": "getOrder",
        "parameters": {"orderId": "$request.query.ord%2Fid"},
    }
    doc = make_doc(link, targets=targets)
    ctx = {
        "request": {"query": {"ord/id": "q7"}},
        "response": {"statusCode": 201},
    }
    response = plan(doc, context=ctx)
    assert response.status_code == 200
    assert response.json()["parameters"]["path"] == {"orderId": "q7"}


def test_boolean_and_number_values_rendered_as_json_values():
    targets = target_get(
        parameters=[
            {"name": "orderId", "in": "path", "required": True},
            {"name": "flag", "in": "query"},
        ]
    )
    link = {
        "operationId": "getOrder",
        "parameters": {
            "orderId": "$response.body#/n",
            "flag": "$response.body#/b",
        },
    }
    doc = make_doc(link, targets=targets)
    response = plan(doc, ctx_obj({"n": 12, "b": True}))
    assert response.status_code == 200
    grouped = response.json()["parameters"]
    assert grouped["path"] == {"orderId": 12}
    assert grouped["query"] == {"flag": True}


def test_mappings_sorted_by_in_and_name():
    targets = target_get(
        parameters=[
            {"name": "orderId", "in": "path", "required": True},
            {"name": "b", "in": "query"},
            {"name": "a", "in": "query"},
        ]
    )
    link = {
        "operationId": "getOrder",
        "parameters": {
            "b": "$response.body#/b",
            "orderId": "$response.body#/id",
            "a": "$response.body#/a",
        },
    }
    doc = make_doc(link, targets=targets)
    response = plan(doc, ctx_obj({"id": "x", "a": 1, "b": 2}))
    assert response.status_code == 200
    keys = [(m["in"], m["name"]) for m in response.json()["mappings"]]
    assert keys == [("path", "orderId"), ("query", "a"), ("query", "b")]


def test_optional_unassigned_parameters_are_omitted():
    targets = target_get(
        parameters=[
            {"name": "orderId", "in": "path", "required": True},
            {"name": "expand", "in": "query"},
        ]
    )
    doc = make_doc(targets=targets)
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 200
    assert response.json()["parameters"]["query"] == {}


def test_selector_and_path_escaping_in_pointers():
    doc = make_doc(
        targets=target_get(path="/orders/{orderId}"),
        source_path="/orders",
    )
    response = plan(
        doc, selector="post /orders", context=ctx_obj({"id": "x"})
    )
    assert response.status_code == 200
    assert response.json()["target"]["operation_pointer"] == (
        "/paths/~1orders~1{orderId}/get"
    )


# --------------------------------------------------------------------------- #
# 400 errors
# --------------------------------------------------------------------------- #


def test_invalid_document_shape():
    response = plan({"openapi": "2.0", "paths": {}}, ctx_obj())
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_openapi_document"

    response = plan({}, ctx_obj())
    assert response.status_code == 400


def test_invalid_selector():
    doc = make_doc()
    response = plan(doc, selector="GARBAGE", context=ctx_obj({"id": "x"}))
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_selector"

    response = plan(doc, selector="GET /missing", context=ctx_obj({"id": "x"}))
    assert response.status_code == 400

    response = plan(doc, selector="DELETE /orders", context=ctx_obj({"id": "x"}))
    assert response.status_code == 400


def test_empty_link_name_is_400():
    doc = make_doc()
    response = plan(doc, link_name="", context=ctx_obj())
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_link_request"


def test_malformed_context_envelope_is_400():
    doc = make_doc()
    response = client.post(
        "/links/plan",
        json={
            "document": doc,
            "selector": "POST /orders",
            "link_name": "getOrder",
            "context": {"response": "bad"},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_link_request"

    response = client.post(
        "/links/plan",
        json={
            "document": doc,
            "selector": "POST /orders",
            "link_name": "getOrder",
            "context": {"wat": {}},
        },
    )
    assert response.status_code == 400

    response = client.post(
        "/links/plan",
        json={
            "document": doc,
            "selector": "POST /orders",
            "link_name": "getOrder",
            "context": {"response": {"statusCode": True}},
        },
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# 422 errors: response selection / link resolution
# --------------------------------------------------------------------------- #


def test_missing_status_code_422():
    doc = make_doc()
    response = plan(doc, {"response": {"body": {"id": "x"}}})
    assert response.status_code == 422
    (error,) = response.json()["errors"]
    assert error["code"] == "missing_status_code"
    assert error["pointer"] == "/context/response/statusCode"


def test_invalid_status_code_422():
    doc = make_doc()
    for bad in (999, 20, "abc", 600):
        response = plan(doc, ctx_obj({}, status_code=bad))
        assert response.status_code == 422
        assert response.json()["errors"][0]["code"] == "invalid_status"


def test_response_not_declared_422():
    doc = make_doc(responses={"201": {"description": "created"}})
    response = plan(doc, ctx_obj({}, status_code=404))
    assert response.status_code == 422
    error = response.json()["errors"][0]
    assert error["code"] == "response_not_declared"
    assert error["status_code"] == "404"
    assert error["pointer"] == "/paths/~1orders/post/responses"


def test_responses_member_not_object_422():
    doc = make_doc()
    doc["paths"]["/orders"]["post"]["responses"] = []
    response = plan(doc, ctx_obj())
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "invalid_response"


def test_unknown_link_name_422():
    doc = make_doc()
    response = plan(doc, link_name="missing", context=ctx_obj({"id": "x"}))
    assert response.status_code == 422
    error = response.json()["errors"][0]
    assert error["code"] == "unknown_link"
    assert error["name"] == "missing"
    assert error["pointer"] == (
        "/paths/~1orders/post/responses/201/links"
    )


def test_unknown_link_through_ref_points_to_components():
    components = {
        "responses": {"Created": {"description": "created", "links": {}}}
    }
    doc = make_doc(
        responses={"201": {"$ref": "#/components/responses/Created"}},
        components=components,
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    error = response.json()["errors"][0]
    assert error["code"] == "unknown_link"
    assert error["pointer"] == "/components/responses/Created/links"


def test_link_must_have_exactly_one_target_selector():
    # Neither.
    doc = make_doc(link={"parameters": {}})
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "invalid_link"

    # Both.
    link = {
        "operationId": "getOrder",
        "operationRef": "#/paths/~1orders~1{orderId}/get",
        "parameters": {"orderId": "$response.body#/id"},
    }
    doc = make_doc(link)
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "invalid_link"


def test_unknown_operation_id_422():
    doc = make_doc(link={"operationId": "ghost"})
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    error = response.json()["errors"][0]
    assert error["code"] == "invalid_target"
    assert error["ref"] == "ghost"
    assert error["pointer"].endswith("/operationId")


def test_duplicate_operation_id_is_ambiguous():
    doc = make_doc(
        target_extra_paths={
            "/other": {
                "get": {
                    "operationId": "getOrder",
                    "responses": {"200": {"description": "x"}},
                }
            }
        }
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    error = response.json()["errors"][0]
    assert error["code"] == "ambiguous_operation"
    assert error["ref"] == "getOrder"


def test_invalid_operation_ref_forms():
    cases = [
        "https://example.com/api.json#/paths/~1orders/get",
        "#/components/schemas/X",
        "#/paths/~1orders/delete",
        "#/paths/~1missing/get",
        "#/info",
        "file.json#/x",
    ]
    for ref in cases:
        doc = make_doc(
            link={
                "operationRef": ref,
                "parameters": {"orderId": "$response.body#/id"},
            }
        )
        response = plan(doc, ctx_obj({"id": "x"}))
        assert response.status_code == 422, ref
        code = response.json()["errors"][0]["code"]
        assert code in ("invalid_target", "unresolvable_reference"), ref


# --------------------------------------------------------------------------- #
# 422 errors: expressions / parameter mapping
# --------------------------------------------------------------------------- #


def test_missing_expression_value_422():
    # JSON Pointer cannot reach the value.
    doc = make_doc()
    response = plan(doc, ctx_obj({}))
    assert response.status_code == 422
    errors = response.json()["errors"]
    codes = {e["code"] for e in errors}
    assert "missing_value" in codes
    error = next(e for e in errors if e["code"] == "missing_value")
    assert error["expression"] == "$response.body#/id"
    assert error["context_pointer"] == "/context/response/body/id"
    assert error["pointer"].endswith("/parameters/orderId")


def test_missing_request_section_422():
    targets = target_get(
        parameters=[{"name": "orderId", "in": "path", "required": True}]
    )
    link = {
        "operationId": "getOrder",
        "parameters": {"orderId": "$request.path.oid"},
    }
    doc = make_doc(link, targets=targets)
    response = plan(doc, {"response": {"statusCode": 201}})
    assert response.status_code == 422
    codes = {e["code"] for e in response.json()["errors"]}
    assert "missing_value" in codes
    error = next(
        e
        for e in response.json()["errors"]
        if e["code"] == "missing_value"
    )
    assert error["expression"] == "$request.path.oid"


def test_array_index_out_of_range_is_missing():
    doc = make_doc(
        link={
            "operationId": "getOrder",
            "parameters": {"orderId": "$response.body#/items/3/href"},
        }
    )
    response = plan(doc, ctx_obj({"items": [{"href": "a"}]}))
    assert response.status_code == 422
    assert "missing_value" in {
        e["code"] for e in response.json()["errors"]
    }


def test_literal_and_template_expressions_rejected():
    for bad in ("literal", "{$response.body#/id}",
                "$request.body#/bad~2", "$foo", "$response.path.x",
                "$request.header.", "{}"):
        doc = make_doc(
            link={
                "operationId": "getOrder",
                "parameters": {"orderId": bad},
            }
        )
        response = plan(doc, ctx_obj({"id": "x"}))
        assert response.status_code == 422, bad
        codes = {e["code"] for e in response.json()["errors"]}
        assert "invalid_expression" in codes, bad


def test_request_body_bad_expression_422():
    targets = target_get(
        method="put",
        parameters=[{"name": "orderId", "in": "path", "required": True}],
        extra={"operationId": "replaceOrder"},
    )
    link = {
        "operationId": "replaceOrder",
        "parameters": {"orderId": "$response.body#/id"},
        "requestBody": "not-an-expression",
    }
    doc = make_doc(link, targets=targets)
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    error = next(
        e
        for e in response.json()["errors"]
        if e["pointer"].endswith("/requestBody")
    )
    assert error["code"] == "invalid_expression"


def test_request_body_missing_value_422():
    targets = target_get(
        method="put",
        parameters=[{"name": "orderId", "in": "path", "required": True}],
        extra={"operationId": "replaceOrder"},
    )
    link = {
        "operationId": "replaceOrder",
        "parameters": {"orderId": "$response.body#/id"},
        "requestBody": "$request.body",
    }
    doc = make_doc(link, targets=targets)
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    assert "missing_value" in {
        e["code"] for e in response.json()["errors"]
    }


def test_unknown_parameter_mapping_422():
    doc = make_doc(
        link={
            "operationId": "getOrder",
            "parameters": {"wat": "$response.body#/id"},
        }
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    error = next(
        e
        for e in response.json()["errors"]
        if e["code"] == "unknown_parameter"
    )
    assert error["name"] == "wat"
    assert error["pointer"].endswith("/parameters/wat")


def test_cross_location_same_name_is_ambiguous():
    targets = target_get(
        parameters=[
            {"name": "token", "in": "path", "required": True},
            {"name": "token", "in": "query"},
        ]
    )
    doc = make_doc(
        link={
            "operationId": "getOrder",
            "parameters": {"token": "$response.body#/id"},
        },
        targets=targets,
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    # The name maps to parameters in two locations: ambiguous_parameter.
    # No value is assigned, but ambiguity alone is reported (the path
    # parameter cannot be singled out once the name is unresolvable).
    codes = {e["code"] for e in response.json()["errors"]}
    assert "ambiguous_parameter" in codes


def test_header_names_differing_only_in_case_in_contract():
    targets = target_get(
        parameters=[
            {"name": "Trace", "in": "header"},
            {"name": "trace", "in": "header"},
        ]
    )
    doc = make_doc(
        link={"operationId": "getOrder"},
        targets=targets,
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "duplicate_parameter"


def test_ambiguous_context_header_422():
    targets = target_get(
        parameters=[
            {"name": "orderId", "in": "path", "required": True},
            {"name": "trace", "in": "header"},
        ]
    )
    link = {
        "operationId": "getOrder",
        "parameters": {
            "orderId": "$response.body#/id",
            "trace": "$request.header.trace",
        },
    }
    doc = make_doc(link, targets=targets)
    ctx = {
        "request": {"header": {"Trace": "a", "trace": "b"}},
        "response": {"statusCode": 201, "body": {"id": "x"}},
    }
    response = plan(doc, context=ctx)
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "ambiguous_header"


def test_required_parameter_not_assigned_422():
    # A required query parameter with no link mapping.
    targets = target_get(
        parameters=[
            {"name": "orderId", "in": "path", "required": True},
            {"name": "mandatory", "in": "query", "required": True},
        ]
    )
    doc = make_doc(targets=targets)
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    error = next(
        e
        for e in response.json()["errors"]
        if e["code"] == "missing_required_value"
    )
    assert error["in"] == "query"
    assert error["name"] == "mandatory"
    assert error["pointer"].endswith("/parameters/1")


def test_operation_level_parameter_overrides_path_level():
    targets = {
        "/orders/{orderId}": {
            "parameters": [
                {"name": "orderId", "in": "path", "required": True,
                 "x-old": True}
            ],
            "get": {
                "operationId": "getOrder",
                "parameters": [
                    {"name": "orderId", "in": "path", "required": True}
                ],
                "responses": {"200": {"description": "ok"}},
            },
        }
    }
    doc = make_doc(targets=targets)
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 200
    mapping = response.json()["mappings"][0]
    # Operation-level definition is the effective parameter location.
    assert mapping["parameter_pointer"] == (
        "/paths/~1orders~1{orderId}/get/parameters/0"
    )


def test_invalid_target_parameter_definition_422():
    targets = target_get(parameters=[{"in": "query"}])
    doc = make_doc(
        link={"operationId": "getOrder"},
        targets=targets,
    )
    response = plan(doc, ctx_obj())
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "invalid_parameter"


def test_parameters_member_not_object_422():
    doc = make_doc(
        link={
            "operationId": "getOrder",
            "parameters": ["not", "an", "object"],
        }
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "invalid_link"


def test_multiple_errors_are_reported_together_and_sorted():
    # bad1/bad2 are unknown target parameters (one error each); the
    # required path parameter is additionally unassigned.
    targets = target_get(
        parameters=[
            {"name": "orderId", "in": "path", "required": True},
            {"name": "known", "in": "query"},
        ]
    )
    doc = make_doc(
        link={
            "operationId": "getOrder",
            "parameters": {
                "known": "$response.body#/also-missing",
                "bad1": "$response.body#/missing",
                "bad2": "$response.body#/missing2",
            },
        },
        targets=targets,
    )
    response = plan(doc, ctx_obj({}))
    assert response.status_code == 422
    errors = response.json()["errors"]
    codes = [e["code"] for e in errors]
    assert codes == sorted(codes)
    code_set = {e["code"] for e in errors}
    assert code_set == {
        "unknown_parameter",
        "missing_value",
        "missing_required_value",
    }
    assert sum(c == "unknown_parameter" for c in codes) == 2


# --------------------------------------------------------------------------- #
# 422 errors: references
# --------------------------------------------------------------------------- #


def test_dangling_response_ref_422():
    doc = make_doc(
        responses={"201": {"$ref": "#/components/responses/Missing"}}
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    error = response.json()["errors"][0]
    assert error["code"] == "unresolvable_reference"
    assert error["ref"] == "#/components/responses/Missing"
    assert error["pointer"] == (
        "/paths/~1orders/post/responses/201"
    )


def test_external_response_ref_422():
    doc = make_doc(
        responses={"201": {"$ref": "https://x.example.com/r.json"}}
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "unresolvable_reference"


def test_dangling_link_ref_422():
    components = {"links": {}}
    doc = make_doc(
        responses={
            "201": {
                "description": "x",
                "links": {
                    "getOrder": {"$ref": "#/components/links/Missing"}
                },
            }
        },
        components=components,
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    error = response.json()["errors"][0]
    assert error["code"] == "unresolvable_reference"
    assert error["ref"] == "#/components/links/Missing"


def test_circular_link_ref_422():
    components = {
        "links": {
            "A": {"$ref": "#/components/links/B"},
            "B": {"$ref": "#/components/links/A"},
        }
    }
    doc = make_doc(
        responses={
            "201": {
                "description": "x",
                "links": {"getOrder": {"$ref": "#/components/links/A"}},
            }
        },
        components=components,
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "unresolvable_reference"


def test_root_ref_rejected():
    doc = make_doc(responses={"201": {"$ref": "#"}})
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "unresolvable_reference"


def test_dangling_target_parameter_ref_422():
    targets = target_get(
        parameters=[{"$ref": "#/components/parameters/Missing"}]
    )
    doc = make_doc(targets=targets)
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    codes = {e["code"] for e in response.json()["errors"]}
    assert "unresolvable_reference" in codes
    ref_error = next(
        e
        for e in response.json()["errors"]
        if e["code"] == "unresolvable_reference"
    )
    assert ref_error["ref"] == "#/components/parameters/Missing"
    assert ref_error["pointer"].endswith("/get/parameters/0")


def test_dangling_target_request_body_ref_422():
    targets = target_get(
        parameters=[{"name": "orderId", "in": "path", "required": True}],
        extra={
            "requestBody": {"$ref": "#/components/requestBodies/Missing"}
        },
    )
    doc = make_doc(targets=targets)
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    error = response.json()["errors"][0]
    assert error["code"] == "unresolvable_reference"
    assert error["pointer"].endswith("/requestBody")


def test_circular_target_path_item_ref_422():
    components = {
        "pathItems": {
            "A": {"$ref": "#/components/pathItems/B"},
            "B": {"$ref": "#/components/pathItems/A"},
        }
    }
    doc = make_doc(openapi="3.1.0", components=components)
    doc["paths"]["/orders/{orderId}"] = {
        "$ref": "#/components/pathItems/A"
    }
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    codes = {e["code"] for e in response.json()["errors"]}
    assert "unresolvable_reference" in codes
    ref_error = next(
        e
        for e in response.json()["errors"]
        if e["code"] == "unresolvable_reference"
    )
    assert "circular" in ref_error["error"]


def test_link_ref_target_non_object_422():
    doc = make_doc(
        responses={
            "201": {
                "description": "x",
                "links": {"getOrder": {"$ref": "#/info"}},
            }
        }
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    # '#/info' resolves to an object that is not a Link Object.
    assert response.json()["errors"][0]["code"] == "invalid_link"
    assert response.json()["errors"][0]["pointer"] == "/info"

    # A reference to a scalar location fails as an unresolvable target.
    doc = make_doc(
        responses={
            "201": {
                "description": "x",
                "links": {
                    "getOrder": {"$ref": "#/info/title"}
                },
            }
        }
    )
    response = plan(doc, ctx_obj({"id": "x"}))
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "unresolvable_reference"


# --------------------------------------------------------------------------- #
# Non-mutation / atomicity
# --------------------------------------------------------------------------- #


def test_input_document_and_context_are_not_modified():
    doc = make_doc(
        responses={
            "201": {
                "description": "x",
                "links": {
                    "getOrder": {
                        "operationId": "getOrder",
                        "parameters": {"orderId": "$response.body#/id"},
                    }
                },
            }
        }
    )
    ctx = {
        "request": {"body": {"nested": {"v": 1}}},
        "response": {"statusCode": 201, "body": {"id": ["x", {"y": 2}]}},
    }
    doc_before = copy.deepcopy(doc)
    ctx_before = copy.deepcopy(ctx)
    response = plan(doc, context=ctx)
    assert response.status_code == 200
    assert doc == doc_before
    assert ctx == ctx_before

    # Output values are copies: mutating them does not touch the context.
    response.json()["parameters"]["path"]["orderId"].append("z")
    assert ctx["response"]["body"]["id"] == ["x", {"y": 2}]


def test_failed_plan_has_no_partial_output():
    doc = make_doc(
        link={
            "operationId": "getOrder",
            "parameters": {"orderId": "$response.body#/missing"},
        }
    )
    response = plan(doc, ctx_obj({}))
    assert response.status_code == 422
    body = response.json()
    assert set(body) == {"error", "message", "errors"}
    assert "parameters" not in body
    assert "target" not in body


def test_order_independence_of_link_parameter_keys():
    targets = target_get(
        parameters=[
            {"name": "orderId", "in": "path", "required": True},
            {"name": "a", "in": "query"},
        ]
    )
    link1 = {
        "operationId": "getOrder",
        "parameters": {
            "a": "$response.body#/a",
            "orderId": "$response.body#/id",
        },
    }
    link2 = {
        "operationId": "getOrder",
        "parameters": {
            "orderId": "$response.body#/id",
            "a": "$response.body#/a",
        },
    }
    r1 = plan(make_doc(link1, targets=targets), ctx_obj({"id": "x", "a": 1}))
    r2 = plan(make_doc(link2, targets=targets), ctx_obj({"id": "x", "a": 1}))
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["mappings"] == r2.json()["mappings"]
    assert r1.json()["parameters"] == r2.json()["parameters"]
