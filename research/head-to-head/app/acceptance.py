"""Grade an expense tracker built from PROMPT.md: python acceptance.py LEDGER_FOLDER [--json OUT]

Each test runs the app (LEDGER_FOLDER/ledger.py, copied to a temporary folder) as a user would, in a fresh data
folder, and checks its output, exit status and ledger.json against PROMPT.md. Standard library only, Python 3.8+.
The app is code a model wrote: read it before you run this, as you would any program."""
import csv
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

TIMEOUT_S = 20
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


class App:
    def __init__(self, src, root):
        self.code = root / "app"
        shutil.copytree(src, self.code, ignore=shutil.ignore_patterns("ledger.json", "__pycache__", "*.csv"))
        self.data = root / "data"
        self.data.mkdir()

    def run(self, *args):
        r = subprocess.run([sys.executable, str(self.code / "ledger.py"), *args], cwd=self.data, capture_output=True,
                           text=True, timeout=TIMEOUT_S)
        return r.returncode, r.stdout, r.stderr

    def ok(self, *args):
        code, out, err = self.run(*args)
        assert code == 0, f"{' '.join(args)!r} exited {code}: {err.strip()[-300:]}"
        return out

    def lines(self, *args):
        return [line.rstrip("\r") for line in self.ok(*args).splitlines() if line.strip()]

    def ledger(self):
        p = self.data / "ledger.json"
        return p.read_bytes() if p.exists() else None

    def fails(self, *args):
        before = self.ledger()
        code, out, err = self.run(*args)
        assert code == 1, f"{' '.join(args)!r} should fail with status 1, exited {code} (stdout {out.strip()[:120]!r})"
        assert err.lstrip().startswith("error:"), f"{' '.join(args)!r}: stderr should start with 'error:': {err[:120]!r}"
        assert self.ledger() == before, f"{' '.join(args)!r} changed ledger.json"


def eq(got, want, what=""):
    assert got == want, f"{what}: got {got!r}, expected {want!r}" if what else f"got {got!r}, expected {want!r}"


def rows(app, *args):
    return [line.split("\t") for line in app.lines("list", *args)]


@test
def add_numbers_expenses_from_1(app):
    eq(app.lines("add", "12.50", "food", "--date", "2026-03-01"), ["added #1"])
    eq(app.lines("add", "3", "coffee", "--date", "2026-03-02"), ["added #2"])


@test
def amounts_keep_whole_cents(app):
    for a in ("12", "12.5", "12.50", "0.10", "0.20"):
        app.ok("add", a, "misc", "--date", "2026-03-05")
    eq([r[2] for r in rows(app)], ["12.00", "12.50", "12.50", "0.10", "0.20"])
    eq(app.lines("summary", "--month", "2026-03"), ["misc\t37.30", "total\t37.30"])


@test
def bad_amounts_are_errors(app):
    app.ok("add", "1", "misc", "--date", "2026-03-05")
    for a in ("0", "-5", "1.234", "abc", "", "1e3"):
        app.fails("add", a, "misc", "--date", "2026-03-05")


@test
def bad_categories_are_errors(app):
    app.ok("add", "1", "misc", "--date", "2026-03-05")
    for c in ("Food", "a b", "x" * 21, "", "a_b"):
        app.fails("add", "5", c, "--date", "2026-03-05")
    app.ok("add", "1", "x" * 20, "--date", "2026-03-05")
    app.ok("add", "1", "home-2", "--date", "2026-03-05")


@test
def bad_dates_are_errors(app):
    app.ok("add", "1", "misc", "--date", "2026-03-05")
    for d in ("2026-02-30", "2026-13-01", "26-01-01", "2026/03/01", "yesterday", "2026-3-1"):
        app.fails("add", "5", "misc", "--date", d)
    app.ok("add", "1", "misc", "--date", "2028-02-29")


@test
def the_date_defaults_to_today(app):
    app.ok("add", "4.20", "lunch")
    eq(rows(app)[0][1], datetime.date.today().isoformat())


@test
def list_is_by_date_then_id_with_tab_fields(app):
    app.ok("add", "5", "food", "--date", "2026-03-09", "--note", "late dinner")
    app.ok("add", "7", "rent", "--date", "2026-03-01")
    app.ok("add", "2.5", "food", "--date", "2026-03-09")
    eq(app.lines("list"), ["2\t2026-03-01\t7.00\trent\t", "1\t2026-03-09\t5.00\tfood\tlate dinner",
                           "3\t2026-03-09\t2.50\tfood\t"])


@test
def list_filters_by_month_and_category(app):
    app.ok("add", "5", "food", "--date", "2026-03-09")
    app.ok("add", "6", "food", "--date", "2026-04-01")
    app.ok("add", "7", "rent", "--date", "2026-03-01")
    eq([r[0] for r in rows(app, "--month", "2026-03")], ["3", "1"])
    eq([r[0] for r in rows(app, "--category", "food")], ["1", "2"])
    eq([r[0] for r in rows(app, "--month", "2026-03", "--category", "food")], ["1"])
    eq(app.ok("list", "--month", "2025-01"), "")
    eq(app.ok("list", "--category", "travel"), "")


@test
def list_with_nothing_recorded_prints_nothing(app):
    eq(app.ok("list"), "")


@test
def delete_and_ids_are_never_reused(app):
    app.ok("add", "5", "food", "--date", "2026-03-09")
    app.ok("add", "6", "food", "--date", "2026-03-10")
    eq(app.lines("delete", "2"), ["deleted #2"])
    eq(app.lines("add", "7", "food", "--date", "2026-03-11"), ["added #3"])
    eq([r[0] for r in rows(app)], ["1", "3"])
    app.fails("delete", "2")
    app.fails("delete", "99")
    app.fails("delete", "x")
    app.ok("delete", "1")
    app.ok("delete", "3")
    eq(app.lines("add", "1", "food", "--date", "2026-03-11"), ["added #4"])


@test
def summary_is_largest_first_then_by_name(app):
    for a, c in (("10", "food"), ("5", "coffee"), ("2.5", "coffee"), ("7.5", "books"), ("30", "rent")):
        app.ok("add", a, c, "--date", "2026-03-15")
    app.ok("add", "100", "rent", "--date", "2026-04-01")
    eq(app.lines("summary", "--month", "2026-03"),
       ["rent\t30.00", "food\t10.00", "books\t7.50", "coffee\t7.50", "total\t55.00"])


@test
def summary_of_an_empty_month(app):
    eq(app.lines("summary", "--month", "2026-03"), ["total\t0.00"])
    app.ok("add", "5", "food", "--date", "2026-04-01")
    eq(app.lines("summary", "--month", "2026-03"), ["total\t0.00"])


@test
def budgets_mark_categories_over_them(app):
    eq(app.lines("budget", "food", "20"), ["budget food 20.00"])
    eq(app.lines("budget", "coffee", "7.50"), ["budget coffee 7.50"])
    for a, c in (("15", "food"), ("6.25", "food"), ("7.50", "coffee"), ("3", "books")):
        app.ok("add", a, c, "--date", "2026-03-15")
    eq(app.lines("summary", "--month", "2026-03"),
       ["food\t21.25\tover by 1.25", "coffee\t7.50", "books\t3.00", "total\t31.75"])
    app.ok("budget", "food", "25")                                            # replaces the earlier budget
    eq(app.lines("summary", "--month", "2026-03")[0], "food\t21.25")
    eq(app.lines("budget", "coffee", "5"), ["budget coffee 5.00"])
    eq(app.lines("summary", "--month", "2026-03")[1], "coffee\t7.50\tover by 2.50")
    eq(app.lines("budget", "coffee", "0"), ["budget coffee removed"])
    eq(app.lines("summary", "--month", "2026-03")[1], "coffee\t7.50")


@test
def a_budget_is_monthly(app):
    app.ok("budget", "food", "10")
    app.ok("add", "8", "food", "--date", "2026-03-15")
    app.ok("add", "8", "food", "--date", "2026-04-15")
    eq(app.lines("summary", "--month", "2026-03"), ["food\t8.00", "total\t8.00"])


@test
def bad_budgets_are_errors(app):
    app.ok("budget", "food", "10")
    for args in (("food", "-1"), ("food", "x"), ("Food", "5"), ("food",), ()):
        app.fails("budget", *args)


@test
def export_writes_csv_by_date_then_id(app):
    app.ok("add", "5", "food", "--date", "2026-03-09", "--note", 'dinner, with "Sam"')
    app.ok("add", "1234.5", "rent", "--date", "2026-03-01")
    app.ok("add", "2", "coffee", "--date", "2026-03-09")
    eq(app.lines("export", "out.csv"), ["exported 3 expenses"])
    with open(app.data / "out.csv", newline="", encoding="utf-8") as f:
        got = list(csv.reader(f))
    eq(got, [["id", "date", "amount", "category", "note"], ["2", "2026-03-01", "1234.50", "rent", ""],
             ["1", "2026-03-09", "5.00", "food", 'dinner, with "Sam"'], ["3", "2026-03-09", "2.00", "coffee", ""]])


@test
def export_of_nothing(app):
    eq(app.lines("export", "none.csv"), ["exported 0 expenses"])
    with open(app.data / "none.csv", newline="", encoding="utf-8") as f:
        eq(list(csv.reader(f)), [["id", "date", "amount", "category", "note"]])


@test
def unknown_or_missing_commands_and_arguments_are_errors(app):
    app.ok("add", "1", "misc", "--date", "2026-03-05")
    for args in ((), ("frobnicate",), ("add",), ("add", "5"), ("delete",), ("summary",), ("export",),
                 ("summary", "--month", "2026-3"), ("summary", "--month", "2026-13"), ("list", "--month", "march"),
                 ("list", "--category", "Bad Cat"), ("add", "5", "food", "--date")):
        app.fails(*args)


@test
def notes_keep_their_spaces(app):
    app.ok("add", "5", "food", "--note", "two  spaces and more", "--date", "2026-03-09")
    eq(rows(app)[0][4], "two  spaces and more")


@test
def data_lives_in_the_current_folder(app):
    app.ok("add", "5", "food", "--date", "2026-03-09")
    assert (app.data / "ledger.json").exists(), "no ledger.json in the current folder"
    json.loads((app.data / "ledger.json").read_text(encoding="utf-8"))


def main(argv):
    if not argv or not (Path(argv[0]) / "ledger.py").exists():
        print(__doc__)
        return 2
    results = []
    for fn in TESTS:
        with tempfile.TemporaryDirectory(prefix="ledger-") as d:
            try:
                fn(App(Path(argv[0]), Path(d)))
                results.append((fn.__name__, None))
            except subprocess.TimeoutExpired as e:
                results.append((fn.__name__, f"timed out after {TIMEOUT_S} s: {e.cmd[2:]}"))
            except AssertionError as e:
                results.append((fn.__name__, str(e)))
            except Exception:  # noqa: BLE001 - a broken test environment, reported per test
                results.append((fn.__name__, traceback.format_exc(limit=1).strip().splitlines()[-1]))
    for name, why in results:
        print(("PASS " if why is None else "FAIL ") + name + ("" if why is None else ": " + why))
    passed = sum(why is None for _, why in results)
    print(f"{passed}/{len(results)} passed")
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(json.dumps(
            {"passed": passed, "total": len(results), "tests": [{"name": n, "ok": w is None, "why": w}
                                                                for n, w in results]}, indent=1), encoding="utf-8")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
