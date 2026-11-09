"""The head-to-head's coding session (research/head-to-head/shop.py): the project as shipped fails the visible tests
it should and nothing else, the bugs fixed and coupons written as SPEC.md says pass every hidden test, and the tools
keep to the project."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "head-to-head"))
import run  # noqa: E402
import shop  # noqa: E402

COUPONS = '''

def find(code, today):
    c = COUPONS.get(code.strip().upper())
    if c is None:
        raise CouponError(f"unknown coupon {code!r}")
    if today > c["expires"]:
        raise CouponError(f"{code!r} expired on {c['expires']}")
    return c


def take_off(coupon, lines, discounts):
    after = {ln.product.sku: ln.subtotal() - discounts.get(ln.product.sku, Money(0)) for ln in lines}
    if coupon["kind"] == "fixed":
        order = sum(after.values(), Money(0))
        return Money(0) if order < coupon["min_order"] else min(coupon["amount"], order)
    return sum((after[ln.product.sku].percent(coupon["pct"]) for ln in lines
                if ln.product.category == coupon["category"]), Money(0))
'''
FIXES = [
    ("money.py", "        return Money(self.cents * pct // 100)",
     "        q, r = divmod(abs(self.cents) * pct, 100)\n        if 2 * r >= 100:\n            q += 1\n"
     "        return Money(q if self.cents >= 0 else -q)"),
    ("rules.py", "free = ln.qty // self.x * self.y", "free = ln.qty // (self.x + self.y) * self.y"),
    ("cart.py", "shipping = shipping_cost(grams, subtotal)", "shipping = shipping_cost(grams, goods)"),
    ("cart.py", "from shipping import shipping_cost", "from shipping import shipping_cost\nimport coupons"),
    ("cart.py", '        raise NotImplementedError("coupons: see SPEC.md")',
     "        self.coupon = coupons.find(code, today)"),
    ("cart.py", "        discounts = sum((d.amount for d in self.discounts()), Money(0))\n        coupon = Money(0)",
     "        ds = self.discounts()\n        discounts = sum((d.amount for d in ds), Money(0))\n"
     "        coupon = coupons.take_off(self.coupon, lines, {d.sku: d.amount for d in ds}) "
     "if self.coupon else Money(0)"),
]


def failed(out):
    return [line[5:].split(":")[0] for line in out.splitlines() if line.startswith("FAIL ")]


class ShopTests(unittest.TestCase):
    def setUp(self):
        self.ws = shop.Workspace(run.unsafe, run.SAFE_IMPORTS)

    def fix(self):
        for path, old, new in FIXES:
            self.assertTrue(self.ws.call("replace_in_file", {"path": path, "old": old, "new": new}).startswith("ok"))
        self.ws.call("write_file", {"path": "coupons.py", "content": shop.FILES["coupons.py"] + COUPONS})

    def test_as_shipped_only_the_bugs_and_coupons_fail(self):
        out = self.ws.call("run_tests", {})
        self.assertIn("RESULT 5 11", out)
        self.assertEqual(failed(out), ["test_percent_rounds_to_the_nearest_cent", "test_buy_two_get_one_free",
                                       "test_free_shipping_is_judged_after_discounts", "test_welcome_coupon",
                                       "test_codes_ignore_case_and_spaces", "test_unknown_and_expired_codes"])
        ok, passed, total, _ = self.ws.grade()
        self.assertEqual((ok, passed, total), (False, 7, 23))

    def test_the_fixed_project_passes_every_hidden_test(self):
        self.fix()
        self.assertIn("RESULT 11 11", self.ws.call("run_tests", {}))
        self.assertEqual(self.ws.grade(), (True, 23, 23, "all passed"))
        self.assertEqual(sorted(self.ws.written), ["cart.py", "coupons.py", "money.py", "rules.py"])

    def test_half_a_fix_is_graded_as_such(self):
        self.ws.call("replace_in_file", {"path": "rules.py", "old": FIXES[1][1], "new": FIXES[1][2]})
        self.ws.call("replace_in_file", {"path": "money.py", "old": FIXES[0][1], "new": "        return Money(round("
                                                                                      "self.cents * pct / 100))"})
        ok, passed, total, detail = self.ws.grade()                 # banker's rounding isn't halves away from zero
        self.assertFalse(ok)
        self.assertIn("test_percent_rounds_halves_away_from_zero", detail)
        self.assertNotIn("test_buy_x_get_y_counts_complete_groups_of_x_plus_y", detail)

    def test_the_tools_keep_to_the_project(self):
        self.assertEqual(self.ws.call("write_file", {"path": "test_cart.py", "content": ""}),
                         "error: test files are read-only")
        self.assertIn("not a file name", self.ws.call("write_file", {"path": "sub/x.py", "content": ""}))
        self.assertIn("occurs 0 times", self.ws.call("replace_in_file", {"path": "cart.py", "old": "zzz", "new": "y"}))
        self.assertIn("no file", self.ws.call("read_file", {"path": "nope.py"}))
        self.assertIn("SPEC.md (49 lines)", self.ws.call("list_files", {}))
        self.assertEqual(shop.tool_result(self.ws, "read_file", "{not json"), "error: the arguments are not valid JSON")
        self.ws.call("write_file", {"path": "helpers.py", "content": "import os\n"})
        self.assertIn("not run: helpers.py imports os", self.ws.call("run_tests", {}))
        self.ws.call("write_file", {"path": "helpers.py", "content": "import datetime\nimport money\n"})
        self.ws.call("write_file", {"path": "cart.py", "content": shop.FILES["cart.py"] + "import helpers\n"})
        self.assertIn("RESULT 5 11", self.ws.call("run_tests", {}))   # the project's own modules may be imported


if __name__ == "__main__":
    unittest.main()
