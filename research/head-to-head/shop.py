"""The head-to-head's coding-session task: a small project the model works on through tools, as a coding agent does.

The project is a shop's pricing code (FILES): SPEC.md says how it should work, test_cart.py tests some of it. It has
three bugs the tests catch (a percentage rounded down, buy-X-get-Y counted in groups of X, free shipping judged
before discounts) and a feature to write (coupons). The model lists, reads, writes and edits files and runs the
tests (TOOLS), turn after turn, so its context grows with what it has read and the test output, as in a real session.
It is graded on HIDDEN_TESTS: the visible tests and the rest of SPEC.md (rounding halves, group edges, shipping bands,
every coupon rule).

Workspace runs the code the way run.py runs a code answer: each .py file the model wrote is checked first (`check`,
run.py's unsafe(), with the project's own modules allowed), then the tests run in their own Python process
(isolated mode, an empty folder, a timeout). Test files are read-only.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_S = 60
MODULES = ("money", "catalog", "rules", "shipping", "coupons", "cart")
IMPORTS = {"datetime"}            # beyond run.py's SAFE_IMPORTS: a coupon's expiry is a date

PROMPT = """You're working in a small Python project: a shop's pricing code. SPEC.md is the specification.

1. Some tests in test_cart.py fail. Find and fix the bugs: the tests and SPEC.md are right, the code is wrong.
2. Coupons aren't written yet. Implement Cart.apply_coupon and what it needs (coupons.py), as SPEC.md describes.

Use the tools to look at the files, change them and run the tests. The project uses only the standard library.
test_cart.py doesn't cover everything in SPEC.md: where they differ in coverage, follow SPEC.md. When you're done,
reply with a short summary of what you changed."""

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "The project's files, with their sizes in lines.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "A file's contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create a file or replace its contents. Test files are read-only.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "replace_in_file",
        "description": "Replace text in a file: `old` must occur exactly once. Test files are read-only.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                                        "new": {"type": "string"}},
                       "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run test_cart.py and report each test.",
        "parameters": {"type": "object", "properties": {}}}},
]

FILES = {}
FILES["SPEC.md"] = """# Shop pricing

All amounts are Money (whole cents). Cart.totals() works out an order in this order:

1. subtotal: each line's unit price times its quantity, summed.
2. discounts: the automatic discounts (the cart's rules). A line gets at most one: when several rules discount
   the same line, only the largest applies.
3. coupon: what the cart's coupon takes off, if it has one (see Coupons).
4. goods: subtotal - discounts - coupon.
5. shipping: by the order's weight (see Shipping); free when goods is at least 75.00.
6. tax: 8% of goods (shipping isn't taxed).
7. total: goods + shipping + tax.

totals() returns a dict with the keys subtotal, discounts, coupon, goods, shipping, tax and total, each Money.

## Money

Percentages are whole numbers. A percentage of an amount is rounded to the nearest cent, halves away from zero:
15% of 19.99 is 3.00 (2.9985), 10% of 0.05 is 0.01 (0.005), 10% of -0.05 is -0.01.

## Discounts

- PercentOff(category, pct): pct% off each line of that category, each line rounded on its own.
- BuyXGetY(sku, x, y): in every complete group of x + y units of the product, y are free. Buy 2 get 1 free on 7
  units: two complete groups of three, so 2 units are free.
- BulkPrice(sku, tiers): tiers are (min_qty, unit price); with at least min_qty units, every unit costs the price
  of the highest tier reached, when that is below the list price.

## Shipping

By the order's total weight: up to 500 g 4.99, up to 2,000 g 8.99, up to 5,000 g 14.99; over 5,000 g, 14.99 plus
3.00 for each started kilogram over 5,000 g. An empty order ships for nothing. Shipping is free when goods (after
the discounts and the coupon, before tax) is at least 75.00.

## Coupons

Cart.apply_coupon(code, today) puts a coupon from coupons.COUPONS on the cart; today is an ISO date
("2026-11-08").

- Codes match without regard to case or surrounding spaces: " welcome5 " is WELCOME5.
- An unknown code raises coupons.CouponError, and so does an expired one: a coupon is valid through its expiry
  date. A code that raises leaves the cart's coupon as it was.
- A cart has at most one coupon: applying another replaces it.
- fixed: takes `amount` off when the order after its discounts (subtotal - discounts) is at least `min_order`;
  otherwise it takes nothing off, but stays on the cart (the order may still grow). It never takes off more than
  the order after its discounts.
- category: `pct`% off each line of that category, on what the line costs after its automatic discount, each line
  rounded on its own as in Money.
- What a coupon takes off follows the cart as it is when totals() is called, not as it was when it was applied.
"""

FILES["money.py"] = '''"""Money in whole cents: never floats (SPEC.md, "Money")."""


class Money:
    __slots__ = ("cents",)

    def __init__(self, cents=0):
        if not isinstance(cents, int) or isinstance(cents, bool):
            raise TypeError(f"Money takes whole cents, not {cents!r}")
        self.cents = cents

    @classmethod
    def of(cls, text):
        """Money.of("12.34") is 1234 cents; Money.of("5") is 500."""
        whole, _, frac = text.strip().partition(".")
        if len(frac) > 2:
            raise ValueError(f"more than two decimals: {text!r}")
        sign = -1 if whole.startswith("-") else 1
        return cls(sign * (abs(int(whole)) * 100 + int((frac + "00")[:2])))

    def __add__(self, other):
        return Money(self.cents + other.cents)

    def __sub__(self, other):
        return Money(self.cents - other.cents)

    def times(self, n):
        return Money(self.cents * n)

    def percent(self, pct):
        """pct percent of this amount (pct a whole number), to the cent."""
        return Money(self.cents * pct // 100)

    def __eq__(self, other):
        return isinstance(other, Money) and self.cents == other.cents

    def __lt__(self, other):
        return self.cents < other.cents

    def __le__(self, other):
        return self.cents <= other.cents

    def __gt__(self, other):
        return self.cents > other.cents

    def __ge__(self, other):
        return self.cents >= other.cents

    def __hash__(self):
        return hash(self.cents)

    def __repr__(self):
        return f"Money({self.cents})"

    def __str__(self):
        sign = "-" if self.cents < 0 else ""
        return f"{sign}{abs(self.cents) // 100}.{abs(self.cents) % 100:02d}"
'''

FILES["catalog.py"] = '''"""The products the shop sells."""
from money import Money


class UnknownSku(KeyError):
    pass


class Product:
    def __init__(self, sku, name, price, weight_g, category):
        if not isinstance(price, Money):
            raise TypeError("price must be Money")
        self.sku, self.name, self.price, self.weight_g, self.category = sku, name, price, weight_g, category

    def __repr__(self):
        return f"Product({self.sku!r}, {self.name!r}, {self.price!r})"


class Catalog:
    def __init__(self, products=()):
        self._by_sku = {}
        for p in products:
            self.add(p)

    def add(self, product):
        if product.sku in self._by_sku:
            raise ValueError(f"duplicate sku {product.sku!r}")
        self._by_sku[product.sku] = product

    def get(self, sku):
        try:
            return self._by_sku[sku]
        except KeyError:
            raise UnknownSku(sku) from None

    def __contains__(self, sku):
        return sku in self._by_sku
'''

FILES["rules.py"] = '''"""Automatic discounts (SPEC.md, "Discounts"). A rule's apply(lines) returns the discounts it gives, at most one a
line; the cart keeps the largest for each line."""


class Discount:
    def __init__(self, sku, label, amount):
        self.sku, self.label, self.amount = sku, label, amount

    def __repr__(self):
        return f"Discount({self.sku!r}, {self.label!r}, {self.amount!r})"


class PercentOff:
    """pct percent off each line of a category."""

    def __init__(self, category, pct):
        self.category, self.pct = category, pct

    def apply(self, lines):
        return [Discount(ln.product.sku, f"{self.pct}% off {self.category}", ln.subtotal().percent(self.pct))
                for ln in lines if ln.product.category == self.category]


class BuyXGetY:
    """Buy x of a product, get y more of it free."""

    def __init__(self, sku, x, y):
        self.sku, self.x, self.y = sku, x, y

    def apply(self, lines):
        for ln in lines:
            if ln.product.sku == self.sku:
                free = ln.qty // self.x * self.y
                if free:
                    return [Discount(self.sku, f"buy {self.x} get {self.y} free", ln.product.price.times(free))]
        return []


class BulkPrice:
    """Tiered unit prices: from min_qty units on, every unit costs that tier's price."""

    def __init__(self, sku, tiers):
        self.sku, self.tiers = sku, sorted(tiers, key=lambda t: t[0])

    def apply(self, lines):
        for ln in lines:
            if ln.product.sku != self.sku:
                continue
            price = None
            for min_qty, unit in self.tiers:
                if ln.qty >= min_qty:
                    price = unit
            if price is not None and price < ln.product.price:
                return [Discount(self.sku, "bulk price", (ln.product.price - price).times(ln.qty))]
        return []
'''

FILES["shipping.py"] = '''"""Shipping (SPEC.md, "Shipping")."""
from money import Money

BANDS = [(500, Money(499)), (2000, Money(899)), (5000, Money(1499))]    # up to this many grams: this price
PER_KG_OVER = Money(300)        # over the last band: its price plus this for each started kilogram over it
FREE_FROM = Money(7500)         # goods of at least this much ship free


def weight_cost(grams):
    if grams <= 0:
        return Money(0)
    for limit, price in BANDS:
        if grams <= limit:
            return price
    started_kg = -(-(grams - BANDS[-1][0]) // 1000)
    return BANDS[-1][1] + PER_KG_OVER.times(started_kg)


def shipping_cost(grams, goods):
    """goods: the order after its discounts and coupon, before tax."""
    if goods >= FREE_FROM:
        return Money(0)
    return weight_cost(grams)
'''

FILES["coupons.py"] = '''"""Coupon codes (SPEC.md, "Coupons"). Not written yet."""
from money import Money


class CouponError(ValueError):
    pass


# code -> the coupon; a coupon is valid through its expiry date
COUPONS = {
    "WELCOME5": {"kind": "fixed", "amount": Money(500), "min_order": Money(2500), "expires": "2030-12-31"},
    "BIG15": {"kind": "fixed", "amount": Money(1500), "min_order": Money(6000), "expires": "2030-12-31"},
    "THANKS10": {"kind": "fixed", "amount": Money(1000), "min_order": Money(0), "expires": "2030-12-31"},
    "TEA20": {"kind": "category", "category": "tea", "pct": 20, "expires": "2030-12-31"},
    "SPRING10": {"kind": "fixed", "amount": Money(1000), "min_order": Money(3000), "expires": "2026-05-31"},
}
'''

FILES["cart.py"] = '''"""A shopping cart and its totals (SPEC.md)."""
from money import Money
from shipping import shipping_cost

TAX_PCT = 8


class Line:
    def __init__(self, product, qty):
        self.product, self.qty = product, qty

    def subtotal(self):
        return self.product.price.times(self.qty)


class Cart:
    def __init__(self, catalog, rules=()):
        self.catalog, self.rules = catalog, list(rules)
        self._qty = {}
        self.coupon = None

    def add(self, sku, qty=1):
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.catalog.get(sku)                       # UnknownSku for a product the shop doesn't sell
        self._qty[sku] = self._qty.get(sku, 0) + qty

    def remove(self, sku, qty=None):
        """Take qty units of sku out of the cart (all of them without qty); KeyError if it isn't in the cart."""
        if sku not in self._qty:
            raise KeyError(sku)
        left = 0 if qty is None else self._qty[sku] - qty
        if left > 0:
            self._qty[sku] = left
        else:
            del self._qty[sku]

    def lines(self):
        return [Line(self.catalog.get(sku), qty) for sku, qty in sorted(self._qty.items())]

    def discounts(self):
        """The automatic discounts: for each line, the largest any rule gives it."""
        lines, best = self.lines(), {}
        for rule in self.rules:
            for d in rule.apply(lines):
                if d.sku not in best or d.amount > best[d.sku].amount:
                    best[d.sku] = d
        return [best[sku] for sku in sorted(best)]

    def apply_coupon(self, code, today):
        raise NotImplementedError("coupons: see SPEC.md")

    def totals(self):
        lines = self.lines()
        subtotal = sum((ln.subtotal() for ln in lines), Money(0))
        discounts = sum((d.amount for d in self.discounts()), Money(0))
        coupon = Money(0)
        goods = subtotal - discounts - coupon
        grams = sum(ln.product.weight_g * ln.qty for ln in lines)
        shipping = shipping_cost(grams, subtotal)
        tax = goods.percent(TAX_PCT)
        return {"subtotal": subtotal, "discounts": discounts, "coupon": coupon, "goods": goods,
                "shipping": shipping, "tax": tax, "total": goods + shipping + tax}
'''

_TEST_HEAD = '''from money import Money
from catalog import Catalog, Product, UnknownSku
from rules import PercentOff, BuyXGetY, BulkPrice
from cart import Cart
from coupons import CouponError


def m(text):
    return Money.of(text)


def eq(got, want):
    assert got == want, f"got {got!r}, expected {want!r}"


def shop():
    return Catalog([
        Product("TEA-GRN", "Green tea", m("6.50"), 120, "tea"),
        Product("TEA-BLK", "Black tea", m("5.25"), 120, "tea"),
        Product("MUG", "Mug", m("12.00"), 350, "kitchen"),
        Product("KETTLE", "Kettle", m("40.00"), 1400, "kitchen"),
        Product("HONEY", "Honey", m("8.75"), 500, "pantry"),
        Product("BISC", "Biscuits", m("3.20"), 200, "pantry"),
    ])


def cart(*items, rules=()):
    c = Cart(shop(), rules)
    for sku, qty in items:
        c.add(sku, qty)
    return c


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")
'''

FILES["test_cart.py"] = _TEST_HEAD + '''

def test_money_parses_and_prints():
    eq(m("12.34").cents, 1234)
    eq(m("5").cents, 500)
    eq(str(m("-0.50")), "-0.50")


def test_percent_rounds_to_the_nearest_cent():
    eq(m("19.99").percent(15), m("3.00"))


def test_totals_without_rules():
    t = cart(("MUG", 2)).totals()
    eq((t["subtotal"], t["discounts"], t["coupon"], t["goods"]), (m("24.00"), m("0"), m("0"), m("24.00")))
    eq((t["shipping"], t["tax"], t["total"]), (m("8.99"), m("1.92"), m("34.91")))


def test_percent_off_a_category():
    t = cart(("TEA-GRN", 2), ("MUG", 1), rules=[PercentOff("tea", 10)]).totals()
    eq(t["discounts"], m("1.30"))


def test_buy_two_get_one_free():
    eq(cart(("BISC", 3), rules=[BuyXGetY("BISC", 2, 1)]).totals()["discounts"], m("3.20"))
    eq(cart(("BISC", 4), rules=[BuyXGetY("BISC", 2, 1)]).totals()["discounts"], m("3.20"))


def test_a_line_gets_only_its_largest_discount():
    rules = [PercentOff("pantry", 10), BulkPrice("HONEY", [(3, m("7.50"))])]
    eq(cart(("HONEY", 4), rules=rules).totals()["discounts"], m("5.00"))


def test_free_shipping_is_judged_after_discounts():
    t = cart(("KETTLE", 2), rules=[PercentOff("kitchen", 10)]).totals()
    eq((t["goods"], t["shipping"]), (m("72.00"), m("14.99")))


def test_unknown_products():
    raises(UnknownSku, cart().add, "NOPE")


def test_welcome_coupon():
    c = cart(("MUG", 3))
    c.apply_coupon("WELCOME5", "2026-11-08")
    t = c.totals()
    eq((t["coupon"], t["goods"], t["tax"]), (m("5.00"), m("31.00"), m("2.48")))


def test_codes_ignore_case_and_spaces():
    c = cart(("MUG", 3))
    c.apply_coupon("  welcome5 ", "2026-11-08")
    eq(c.totals()["coupon"], m("5.00"))


def test_unknown_and_expired_codes():
    raises(CouponError, cart(("MUG", 3)).apply_coupon, "NOPE", "2026-11-08")
    raises(CouponError, cart(("MUG", 3)).apply_coupon, "SPRING10", "2026-11-08")
'''

HIDDEN_TESTS = _TEST_HEAD + '''

def test_percent_rounds_halves_away_from_zero():
    eq(m("19.99").percent(15), m("3.00"))
    eq(m("0.05").percent(10), m("0.01"))
    eq(m("0.05").percent(50), m("0.03"))
    eq(m("-0.05").percent(10), m("-0.01"))
    eq(m("-19.99").percent(15), m("-3.00"))
    eq(m("12.50").percent(8), m("1.00"))
    eq(m("1.99").percent(8), m("0.16"))


def test_tax_is_rounded_to_the_nearest_cent():
    t = cart(("BISC", 1)).totals()
    eq((t["tax"], t["total"]), (m("0.26"), m("8.45")))


def test_totals_without_rules():
    t = cart(("MUG", 2)).totals()
    eq((t["subtotal"], t["discounts"], t["coupon"], t["goods"]), (m("24.00"), m("0"), m("0"), m("24.00")))
    eq((t["shipping"], t["tax"], t["total"]), (m("8.99"), m("1.92"), m("34.91")))


def test_an_empty_cart():
    t = cart().totals()
    eq(sorted(t), ["coupon", "discounts", "goods", "shipping", "subtotal", "tax", "total"])
    eq(t["total"], m("0"))


def test_percent_off_rounds_each_line():
    t = cart(("TEA-GRN", 2), ("TEA-BLK", 1), ("MUG", 1), rules=[PercentOff("tea", 10)]).totals()
    eq(t["discounts"], m("1.83"))                   # 1.30 + 0.53 (0.525)


def test_buy_x_get_y_counts_complete_groups_of_x_plus_y():
    for qty, free in ((2, 0), (3, 1), (4, 1), (5, 1), (6, 2), (7, 2)):
        eq(cart(("BISC", qty), rules=[BuyXGetY("BISC", 2, 1)]).totals()["discounts"], m("3.20").times(free))
    eq(cart(("BISC", 9), rules=[BuyXGetY("BISC", 3, 2)]).totals()["discounts"], m("6.40"))
    eq(cart(("BISC", 10), rules=[BuyXGetY("BISC", 3, 2)]).totals()["discounts"], m("12.80"))


def test_bulk_price_takes_the_highest_tier_reached():
    rules = [BulkPrice("HONEY", [(10, m("6.00")), (3, m("7.50"))])]
    eq(cart(("HONEY", 2), rules=rules).totals()["discounts"], m("0"))
    eq(cart(("HONEY", 3), rules=rules).totals()["discounts"], m("3.75"))
    eq(cart(("HONEY", 10), rules=rules).totals()["discounts"], m("27.50"))


def test_a_line_gets_only_its_largest_discount():
    rules = [PercentOff("pantry", 10), BulkPrice("HONEY", [(3, m("7.50"))])]
    eq(cart(("HONEY", 4), rules=rules).totals()["discounts"], m("5.00"))
    eq(cart(("HONEY", 2), ("BISC", 1), rules=rules).totals()["discounts"], m("2.07"))   # 1.75 + 0.32


def test_shipping_bands():
    for grams, cost in ((120, "4.99"), (500, "4.99"), (620, "8.99"), (2000, "8.99"), (2120, "14.99"),
                        (5000, "14.99"), (5600, "17.99"), (7000, "20.99"), (7120, "23.99")):
        import shipping
        eq(shipping.weight_cost(grams), m(cost))


def test_free_shipping_is_judged_after_discounts():
    t = cart(("KETTLE", 2), rules=[PercentOff("kitchen", 10)]).totals()
    eq((t["goods"], t["shipping"], t["total"]), (m("72.00"), m("14.99"), m("92.75")))
    t = cart(("KETTLE", 2)).totals()
    eq((t["shipping"], t["total"]), (m("0"), m("86.40")))


def test_free_shipping_from_exactly_75():
    c = cart(("KETTLE", 2))
    c.apply_coupon("WELCOME5", "2026-11-08")                            # 80.00 - 5.00
    t = c.totals()
    eq((t["goods"], t["shipping"]), (m("75.00"), m("0")))
    t = cart(("KETTLE", 1), ("MUG", 2), ("TEA-GRN", 1), ("BISC", 1)).totals()     # 73.70, 2,420 g
    eq((t["goods"], t["shipping"]), (m("73.70"), m("14.99")))


def test_cart_remove():
    c = cart(("MUG", 3), ("BISC", 1))
    c.remove("MUG", 2)
    c.remove("BISC")
    eq(c.totals()["subtotal"], m("12.00"))
    raises(KeyError, c.remove, "BISC")


def test_unknown_products():
    raises(UnknownSku, cart().add, "NOPE")


def test_welcome_coupon():
    c = cart(("MUG", 3))
    c.apply_coupon("WELCOME5", "2026-11-08")
    t = c.totals()
    eq((t["coupon"], t["goods"], t["tax"]), (m("5.00"), m("31.00"), m("2.48")))
    eq((t["shipping"], t["total"]), (m("8.99"), m("42.47")))


def test_codes_ignore_case_and_spaces():
    c = cart(("MUG", 3))
    c.apply_coupon("  welcome5 ", "2026-11-08")
    eq(c.totals()["coupon"], m("5.00"))


def test_unknown_codes_raise_and_leave_the_coupon_as_it_was():
    c = cart(("MUG", 3))
    raises(CouponError, c.apply_coupon, "NOPE", "2026-11-08")
    eq(c.totals()["coupon"], m("0"))
    c.apply_coupon("WELCOME5", "2026-11-08")
    raises(CouponError, c.apply_coupon, "NOPE", "2026-11-08")
    raises(CouponError, c.apply_coupon, "SPRING10", "2026-11-08")
    eq(c.totals()["coupon"], m("5.00"))


def test_a_coupon_is_valid_through_its_expiry_date():
    c = cart(("MUG", 3))
    c.apply_coupon("SPRING10", "2026-05-31")
    eq(c.totals()["coupon"], m("10.00"))
    raises(CouponError, cart(("MUG", 3)).apply_coupon, "SPRING10", "2026-06-01")
    raises(CouponError, cart(("MUG", 3)).apply_coupon, "spring10", "2027-01-15")


def test_a_fixed_coupon_needs_its_minimum_after_discounts():
    c = cart(("MUG", 2), ("BISC", 1), rules=[PercentOff("pantry", 10)])      # 27.20 - 0.32 = 26.88
    c.apply_coupon("WELCOME5", "2026-11-08")
    eq(c.totals()["coupon"], m("5.00"))
    c = cart(("MUG", 2), ("TEA-BLK", 1), rules=[PercentOff("tea", 10)])      # 29.25 - 0.53 = 28.72
    c.apply_coupon("WELCOME5", "2026-11-08")
    eq(c.totals()["coupon"], m("5.00"))
    c = cart(("MUG", 2), rules=[PercentOff("kitchen", 10)])                  # 24.00 - 2.40 = 21.60
    c.apply_coupon("WELCOME5", "2026-11-08")
    eq(c.totals()["coupon"], m("0"))
    c = cart(("MUG", 2), ("TEA-GRN", 1), rules=[PercentOff("tea", 100)])     # 30.50 - 6.50 = 24.00
    c.apply_coupon("WELCOME5", "2026-11-08")
    eq(c.totals()["coupon"], m("0"))


def test_a_coupon_follows_the_cart_as_it_is_now():
    c = cart(("MUG", 2))                                                     # 24.00: below the minimum
    c.apply_coupon("WELCOME5", "2026-11-08")
    eq(c.totals()["coupon"], m("0"))
    c.add("BISC", 1)                                                         # 27.20
    eq(c.totals()["coupon"], m("5.00"))
    c.remove("MUG", 1)                                                       # 15.20
    eq(c.totals()["coupon"], m("0"))


def test_a_fixed_coupon_never_takes_off_more_than_the_order():
    c = cart(("BISC", 1))
    c.apply_coupon("THANKS10", "2026-11-08")
    t = c.totals()
    eq((t["coupon"], t["goods"], t["tax"], t["shipping"], t["total"]), (m("3.20"), m("0"), m("0"), m("4.99"),
                                                                         m("4.99")))
    c = cart(("BISC", 3), rules=[BuyXGetY("BISC", 2, 1)])                    # 9.60 - 3.20 = 6.40
    c.apply_coupon("THANKS10", "2026-11-08")
    eq(c.totals()["coupon"], m("6.40"))


def test_a_category_coupon_is_on_each_line_after_its_discount():
    c = cart(("TEA-GRN", 2), ("TEA-BLK", 1), ("MUG", 1), rules=[PercentOff("tea", 10)])
    c.apply_coupon("TEA20", "2026-11-08")
    t = c.totals()
    eq(t["coupon"], m("3.28"))                      # 20% of 11.70 (2.34) + 20% of 4.72 (0.944)
    eq(t["goods"], m("25.14"))                      # 30.25 - 1.83 - 3.28
    c = cart(("TEA-BLK", 1))
    c.apply_coupon("tea20", "2026-11-08")
    eq(c.totals()["coupon"], m("1.05"))
    c = cart(("MUG", 1))
    c.apply_coupon("TEA20", "2026-11-08")
    eq(c.totals()["coupon"], m("0"))


def test_one_coupon_at_a_time():
    c = cart(("TEA-GRN", 1), ("MUG", 3))
    c.apply_coupon("WELCOME5", "2026-11-08")
    c.apply_coupon("TEA20", "2026-11-08")
    eq(c.totals()["coupon"], m("1.30"))
    c.apply_coupon("WELCOME5", "2026-11-08")
    eq(c.totals()["coupon"], m("5.00"))


def test_a_coupon_can_take_away_free_shipping():
    c = cart(("KETTLE", 2))                                                  # 80.00: free shipping
    c.apply_coupon("BIG15", "2026-11-08")
    t = c.totals()
    eq((t["coupon"], t["goods"], t["shipping"], t["tax"], t["total"]),
       (m("15.00"), m("65.00"), m("14.99"), m("5.20"), m("85.19")))
'''

RUNNER = '''import sys
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
results = []
try:
    mod = __import__(sys.argv[2])
except BaseException as e:
    print(f"ERROR importing {sys.argv[2]}: " + "".join(traceback.format_exception_only(type(e), e)).strip())
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename.startswith(sys.argv[1])]
    if frames:
        print(f"  at {frames[-1].filename[len(sys.argv[1]) + 1:]} line {frames[-1].lineno}: {frames[-1].line}")
    print("RESULT 0 0 import failed")
    sys.exit(0)
for name, fn in list(vars(mod).items()):
    if not name.startswith("test_") or not callable(fn):
        continue
    try:
        fn()
        results.append((name, None))
    except BaseException as e:
        why = "".join(traceback.format_exception_only(type(e), e)).strip()
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename.startswith(sys.argv[1])]
        if frames and not frames[-1].filename.endswith(sys.argv[2] + ".py"):
            f = frames[-1]
            why += f" (at {f.filename[len(sys.argv[1]) + 1:]} line {f.lineno}: {f.line})"
        results.append((name, why))
for name, why in results:
    print(("PASS " + name) if why is None else ("FAIL " + name + ": " + why))
failed = [n for n, w in results if w is not None]
print(f"RESULT {len(results) - len(failed)} {len(results)} {', '.join(failed) or 'all passed'}")
'''


class Workspace:
    """The project as the model has changed it, and the tools it uses on it. check(code, imports) -> why the code
    may not run, or None (run.py's unsafe)."""

    def __init__(self, check, safe_imports):
        self.files = dict(FILES)
        self.check, self.imports = check, set(safe_imports) | IMPORTS | set(MODULES)
        self.calls, self.test_runs, self.written = [], 0, set()

    def call(self, name, args):
        self.calls.append(name)
        path = str(args.get("path", "")).strip().lstrip("./")
        if name == "list_files":
            return "\n".join(f"{p} ({t.count(chr(10))} lines)" for p, t in sorted(self.files.items()))
        if name == "read_file":
            return self.files[path] if path in self.files else f"error: no file {path!r}"
        if name in ("write_file", "replace_in_file"):
            if not re.fullmatch(r"[\w.-]+", path) or path.startswith("."):
                return f"error: {path!r} is not a file name in the project (it has no folders)"
            if path.startswith("test_"):
                return "error: test files are read-only"
            if name == "write_file":
                content = args.get("content")
                if not isinstance(content, str):
                    return "error: content must be a string"
                self.files[path] = content
            else:
                old, new = args.get("old"), args.get("new")
                if not isinstance(old, str) or not isinstance(new, str) or not old:
                    return "error: old and new must be strings, old not empty"
                text = self.files.get(path)
                if text is None:
                    return f"error: no file {path!r}"
                n = text.count(old)
                if n != 1:
                    return f"error: old occurs {n} times in {path}; it must occur exactly once"
                self.files[path] = text.replace(old, new)
            self.written.add(path)
            return f"ok: {path} has {self.files[path].count(chr(10))} lines"
        if name == "run_tests":
            self.test_runs += 1
            return self.run("test_cart", self.files["test_cart.py"])[2]
        return f"error: no tool {name!r}"

    def run(self, module, tests):
        """(passed, total, output) of a test module run against the project as it is now."""
        own = {p[:-3] for p in self.files if p.endswith(".py")}         # a module the model added, too
        for p in sorted(self.written):
            if p.endswith(".py"):
                why = self.check(self.files[p], self.imports | own)
                if why:
                    return 0, 0, f"not run: {p} {why} (only the standard library's safe modules and the project's)"
        with tempfile.TemporaryDirectory(prefix="h2h-shop-") as d:
            root = Path(d) / "project"
            root.mkdir()
            for p, text in self.files.items():
                (root / p).write_text(text, encoding="utf-8")
            (root / f"{module}.py").write_text(tests, encoding="utf-8")
            (Path(d) / "_runner.py").write_text(RUNNER, encoding="utf-8")
            env = {k: v for k, v in os.environ.items() if k.upper() in ("SYSTEMROOT", "PATH", "TEMP", "TMP")}
            try:
                r = subprocess.run([sys.executable, "-I", "-S", str(Path(d) / "_runner.py"), str(root), module],
                                   cwd=d, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
            except subprocess.TimeoutExpired:
                return 0, 0, f"the tests ran past {TIMEOUT_S} s"
        out = r.stdout.strip() or ("no output; " + (r.stderr.strip().splitlines() or ["?"])[-1])
        m = re.search(r"^RESULT (\d+) (\d+)", out, re.M)
        return (int(m.group(1)), int(m.group(2)), out) if m else (0, 0, out[-2000:])

    def grade(self):
        """(passed all, passed, total, detail) on HIDDEN_TESTS."""
        passed, total, out = self.run("test_hidden", HIDDEN_TESTS)
        if not total:
            return False, 0, 0, out[:300]
        failed = [line[5:].split(":")[0] for line in out.splitlines() if line.startswith("FAIL ")]
        return passed == total, passed, total, ("all passed" if passed == total else "failed: " + ", ".join(failed))


def tool_result(ws, name, arguments):
    try:
        args = json.loads(arguments or "{}")
    except ValueError:
        return "error: the arguments are not valid JSON"
    return ws.call(name, args if isinstance(args, dict) else {})
