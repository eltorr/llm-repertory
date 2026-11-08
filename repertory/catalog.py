"""The catalog (catalog/catalog.json): the engines and models Repertory offers, and how to set up and run each pair.

load() reads it and checks it against catalog/schema.json (a small validator for the keywords the schema uses, so the
manager needs no extra package) and for what a schema can't express: unique ids, engines that exist, a model format
its engine reads. The lookups are on Catalog; the helpers below take one model entry.

Standard library only, and no import of the rest of the manager: the engine adapter's sizing.py reads the catalog
too, run by run.sh with the system Python inside WSL, where it must not create the manager's settings folder.

A model's `quant` is its short name in the manager: the `quant` setting, bench run specs (QUANT:VRAM:CONTEXT), run.sh's
QUANT and sizing's key. Its setup step is "m" plus that name without dots (m305, m405).
"""
import datetime
import json
import os
import re
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / "catalog" / "catalog.json"
SCHEMA = REPO / "catalog" / "schema.json"
REFERENCE_DIR = REPO / "catalog" / "reference"
STATUSES = ("curated", "failed", "untested")


class CatalogError(ValueError):
    """The catalog (or a schema the validator can't read) is invalid; the message lists what is wrong."""


# ------------------------------------------------------------------------------------------------------------ schema
# The JSON Schema keywords schema.json uses. A keyword outside this set is an error, not silently ignored: extend the
# validator when the schema needs one.
KEYWORDS = {"$schema", "$id", "$ref", "$defs", "title", "description", "default", "type", "required", "properties",
            "additionalProperties", "items", "minItems", "enum", "const", "pattern", "oneOf", "exclusiveMinimum",
            "format"}
TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is(value, kind):
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind not in TYPES:
        raise CatalogError(f"unknown type {kind!r} in the schema")
    return isinstance(value, TYPES[kind])


def _same(a, b):
    """JSON equality: true is not 1."""
    return a == b and isinstance(a, bool) == isinstance(b, bool)


def _resolve(root, ref):
    if not ref.startswith("#/"):
        raise CatalogError(f"only local references are supported, not {ref!r}")
    node = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            raise CatalogError(f"the schema has no {ref}")
        node = node[part]
    return node


def validate(value, schema, root=None, path="$"):
    """What is wrong with `value` against `schema`, as a list of 'path: message' (empty when it is valid)."""
    root = schema if root is None else root
    if not isinstance(schema, dict):
        raise CatalogError(f"schema at {path} is not an object")
    unknown = set(schema) - KEYWORDS
    if unknown:
        raise CatalogError(f"schema at {path} uses keywords the validator doesn't support: {sorted(unknown)}")
    errors = []
    if "$ref" in schema:
        errors += validate(value, _resolve(root, schema["$ref"]), root, path)
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if not any(_is(value, k) for k in kinds):
            return errors + [f"{path}: should be {' or '.join(kinds)}, not {type(value).__name__}"]
    if "const" in schema and not _same(value, schema["const"]):
        errors.append(f"{path}: should be {json.dumps(schema['const'])}")
    if "enum" in schema and not any(_same(value, e) for e in schema["enum"]):
        errors.append(f"{path}: should be one of {', '.join(json.dumps(e) for e in schema['enum'])}")
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} doesn't match {schema['pattern']}")
        if schema.get("format") == "date":
            try:
                datetime.date.fromisoformat(value)
                ok = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))
            except ValueError:
                ok = False
            if not ok:
                errors.append(f"{path}: {value!r} is not a date (YYYY-MM-DD)")
    if _is(value, "number") and "exclusiveMinimum" in schema and not value > schema["exclusiveMinimum"]:
        errors.append(f"{path}: should be more than {schema['exclusiveMinimum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: needs at least {schema['minItems']} item(s)")
        if "items" in schema:
            for i, item in enumerate(value):
                errors += validate(item, schema["items"], root, f"{path}[{i}]")
    if isinstance(value, dict):
        for key in schema.get("required", ()):
            if key not in value:
                errors.append(f"{path}: {key} is missing")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in props:
                errors += validate(item, props[key], root, f"{path}.{key}")
            elif extra is False:
                errors.append(f"{path}: {key} is not allowed here")
            elif isinstance(extra, dict):
                errors += validate(item, extra, root, f"{path}.{key}")
    if "oneOf" in schema:
        results = [validate(value, s, root, path) for s in schema["oneOf"]]
        matched = sum(1 for r in results if not r)
        if matched != 1:
            why = "; ".join(r[0] for r in results if r)
            errors.append(f"{path}: should match exactly one of {len(results)} forms, matches {matched}"
                          + (f" ({why})" if matched == 0 else ""))
    return errors


def _relative(p):
    """A path inside the engine checkout: relative, no '..', plain characters (setup's scripts take it as an argument
    and the checks build file paths from it)."""
    q = PurePosixPath(p)
    return bool(re.fullmatch(r"[A-Za-z0-9._/-]+", p)) and not q.is_absolute() and ".." not in q.parts


def problems(data):
    """What the schema can't check: ids unique, engines that exist, formats they read, paths inside the checkout."""
    out = []
    engines = {}
    for i, e in enumerate(data.get("engines") or []):
        if e["id"] in engines:
            out.append(f"$.engines[{i}]: engine id {e['id']} is used twice")
        engines[e["id"]] = e
        serve = e.get("serve") or {}
        words = serve.get("command", []) + list((serve.get("env") or {}).values())
        used = set(re.findall(r"\{(\w+)\}", " ".join(words)))
        missing = used - set(serve.get("placeholders", []))
        if missing:
            out.append(f"$.engines[{i}].serve: placeholders used but not listed: {sorted(missing)}")
        steps = [s["id"] for s in e.get("build", [])]
        if len(steps) != len(set(steps)):
            out.append(f"$.engines[{i}].build: step ids repeat")
    ids, quants, setups = set(), set(), set()
    for i, m in enumerate(data.get("models") or []):
        where = f"$.models[{i}]"
        for value, seen, what in ((m["id"], ids, "model id"), (m["quant"], quants, "quant"),
                                  (setup_id(m), setups, "setup step id")):
            if value in seen:
                out.append(f"{where}: {what} {value} is used twice")
            seen.add(value)
        for eid in m["engines"]:
            e = engines.get(eid)
            if e is None:
                out.append(f"{where}: engine {eid} is not in the catalog")
            elif m["format"] not in e["supports"]["formats"]:
                out.append(f"{where}: engine {eid} doesn't read the {m['format']} format")
        for key, p in m["paths"].items():
            if not _relative(p):
                out.append(f"{where}.paths.{key}: {p!r} must be a relative path inside the engine checkout")
        steps = [s["id"] for s in m.get("prepare", [])]
        if len(steps) != len(set(steps)):
            out.append(f"{where}.prepare: step ids repeat")
    return out


def check(data, schema):
    """Every error in a catalog document: the schema's, then the ones it can't express."""
    errors = validate(data, schema)
    if not errors:
        errors = problems(data)
    return errors


# ----------------------------------------------------------------------------------------------------------- loading
class Catalog:
    """A loaded, valid catalog. Entries are the document's own dicts: read them, don't change them."""

    def __init__(self, data, path=None):
        self.data, self.path = data, path
        self.engines, self.models = data["engines"], data["models"]

    def engine(self, engine_id):
        for e in self.engines:
            if e["id"] == engine_id:
                return e
        raise KeyError(f"no engine {engine_id!r} in the catalog")

    def find(self, key):
        """A model by its id or its quant, or None."""
        return next((m for m in self.models if key in (m["id"], m["quant"])), None)

    def model(self, key):
        m = self.find(key)
        if m is None:
            raise KeyError(f"no model {key!r} in the catalog (ids: {', '.join(x['id'] for x in self.models)})")
        return m

    def models_for(self, engine_id):
        """The models an engine runs, in catalog order."""
        return [m for m in self.models if engine_id in m["engines"]]

    def engine_of(self, model):
        """The engine that runs a model (its first)."""
        return self.engine(model["engines"][0])


_cache = {}


def load(path=None, schema_path=None):
    """The catalog, checked; CatalogError if it is invalid. Read again when the file changes."""
    path = Path(path or CATALOG)
    st = path.stat()
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(str(path))
    if hit and hit[0] == stamp:
        return hit[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads(Path(schema_path or SCHEMA).read_text(encoding="utf-8"))
    errors = check(data, schema)
    if errors:
        raise CatalogError(f"{path.name}: " + "; ".join(errors[:12]) + (" ..." if len(errors) > 12 else ""))
    cat = Catalog(data, path)
    _cache[str(path)] = (stamp, cat)
    return cat


def save_curation(model_id, curation, path=None, schema_path=None):
    """Write a model's curation result into the catalog file (checked first; the rest of the file is kept as is)."""
    path = Path(path or CATALOG)
    data = json.loads(path.read_text(encoding="utf-8"))
    model = next((m for m in data["models"] if m["id"] == model_id), None)
    if model is None:
        raise KeyError(f"no model {model_id!r} in the catalog")
    model["curation"] = curation
    schema = json.loads(Path(schema_path or SCHEMA).read_text(encoding="utf-8"))
    errors = check(data, schema)
    if errors:
        raise CatalogError("the curation result doesn't fit the schema: " + "; ".join(errors[:12]))
    write_json(path, data, indent=2)


def write_json(path, data, indent=1):
    """JSON as the repository keeps it: UTF-8, LF, a final newline; replaced in one step."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, indent=indent, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------ one model
def status(model):
    """curated, failed or untested (no curation yet counts as untested)."""
    return (model.get("curation") or {}).get("status") or "untested"


def offered(model, show_all=False):
    """Whether setup and the dashboard offer the model: when it is curated; when it is untested and its entry says
    offer_untested (the entries from before curation existed, shown as untested, so nothing changes for them until
    they are curated); never when it failed. show_all (the offer_uncurated setting) offers every one."""
    st = status(model)
    return show_all or st == "curated" or (st == "untested" and bool(model.get("offer_untested")))


def served_name(model):
    """The name clients use for the model (/v1/models): its family."""
    return model["family"]


def setup_id(model):
    """Its setup step's id, also the key setup/check.sh reports it under: m305 for quant 3.05."""
    return "m" + re.sub(r"[^a-z0-9]", "", model["quant"].lower())


def store_name(model):
    """The expert store's name, as the pack script takes it: store/qwen.bin -> qwen."""
    return PurePosixPath(model["paths"]["store"]).stem


def reference_path(model_id):
    """Where a model's regression reference lives (catalog/reference/<model-id>.json)."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,63}", model_id):
        raise ValueError(f"bad model id {model_id!r}")
    return REFERENCE_DIR / f"{model_id}.json"
