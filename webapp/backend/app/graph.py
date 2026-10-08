"""Resolves the drag-and-drop canvas (nodes + edges) into a topologically
ordered DAG. Unlike the original v1 (a single sequential chain), this
supports real branching architectures: any regular layer may fan out to
multiple downstream layers, and exactly two layer types -- `add_merge`
and `concat_merge` -- may fan back in from two branches (e.g. a
residual/skip connection). Everything else must have at most one input
and at least one output, except the single sink (the final layer).
"""

MERGE_TYPES = {"add_merge", "concat_merge"}


class GraphError(Exception):
    """Raised for any structurally invalid canvas graph. Always carries a
    message meant to be shown to the user directly, not a traceback."""


def resolve_dag(nodes: list, edges: list):
    """nodes: [{"id": str, "type": str, "params": dict}, ...]
    edges: [{"source": str, "target": str, "targetHandle": str | None}, ...]

    Returns (order_ids, nodes_by_id, incoming, outgoing, source_id, sink_id):
      - order_ids: node ids in a valid topological order
      - incoming[nid]: list of {"source": id, "target_handle": str | None},
        ordered by target_handle ("in0" before "in1") for merge nodes so
        concat's operand order is deterministic
      - outgoing[nid]: list of downstream node ids
      - source_id: the single node with no incoming edge (feeds the
        dataset's raw input)
      - sink_id: the single node with no outgoing edge (the model's output)

    Raises GraphError (a user-facing message, not a traceback) if the
    graph isn't exactly one DAG with a single source and single sink.
    """
    if not nodes:
        raise GraphError("Add at least one layer to the canvas.")

    nodes_by_id = {n["id"]: n for n in nodes}
    incoming: dict[str, list] = {nid: [] for nid in nodes_by_id}
    outgoing: dict[str, list] = {nid: [] for nid in nodes_by_id}

    for e in edges:
        src, dst = e["source"], e["target"]
        if src not in nodes_by_id or dst not in nodes_by_id:
            raise GraphError("A connection references a layer that no longer exists on the canvas.")
        outgoing[src].append(dst)
        incoming[dst].append({"source": src, "target_handle": e.get("targetHandle")})

    for nid, node in nodes_by_id.items():
        is_merge = node["type"] in MERGE_TYPES
        count = len(incoming[nid])
        if is_merge:
            if count != 2:
                raise GraphError(
                    f"'{node['type']}' layer '{nid}' needs exactly 2 incoming connections "
                    f"(has {count}). Connect two branches into it."
                )
        elif count > 1:
            raise GraphError(
                f"Layer '{nid}' has {count} incoming connections -- only Add/Concat merge "
                f"layers can take more than one input. Use a merge layer to combine branches."
            )

    sources = [nid for nid in nodes_by_id if not incoming[nid]]
    sinks = [nid for nid in nodes_by_id if not outgoing[nid]]
    if len(sources) != 1:
        raise GraphError(
            f"The graph must have exactly one starting layer (no input connection); "
            f"found {len(sources)}. Every layer must eventually trace back to one input."
        )
    if len(sinks) != 1:
        raise GraphError(
            f"The graph must have exactly one final layer (no outgoing connection); "
            f"found {len(sinks)}. Merge every branch back together before the end."
        )

    order = _topological_order(nodes_by_id, incoming, outgoing)

    for nid in nodes_by_id:
        if nodes_by_id[nid]["type"] in MERGE_TYPES:
            incoming[nid].sort(key=lambda e: e["target_handle"] or "")

    return order, nodes_by_id, incoming, outgoing, sources[0], sinks[0]


def _topological_order(nodes_by_id: dict, incoming: dict, outgoing: dict) -> list:
    in_degree = {nid: len(incoming[nid]) for nid in nodes_by_id}
    queue = [nid for nid in nodes_by_id if in_degree[nid] == 0]
    order = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for nxt in outgoing[nid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if len(order) != len(nodes_by_id):
        unresolved = [nid for nid in nodes_by_id if nid not in order]
        raise GraphError(
            f"The layers form a cycle -- cycles aren't supported. Unresolved: "
            f"{', '.join(unresolved)}."
        )
    return order
