Build a command-line expense tracker in Python 3, standard library only, in a new folder named `ledger`. Its entry
point is `ledger.py`, run as `python ledger.py COMMAND ...` from any folder; it keeps its data in `ledger.json` in
the current folder, created on first use. You may split the code into more modules next to `ledger.py`. Test it as
you go, and tell me when it works.

Amounts: a positive number with at most two decimals (`12`, `12.5`, `12.50`), printed with exactly two decimals
(`12.50`). Keep them as whole cents. Categories: lowercase letters, digits and dashes, 1 to 20 characters. Dates:
`YYYY-MM-DD`, a real calendar date. Months: `YYYY-MM`.

Commands:

- `add AMOUNT CATEGORY [--note TEXT] [--date DATE]`: records an expense (today's date if none is given) and prints
  `added #ID`. IDs start at 1 and go up by one; an ID is never reused, even after a delete.
- `list [--month MONTH] [--category CATEGORY]`: one line per matching expense, by date then ID, its fields
  separated by tabs: ID, date, amount, category, note (empty if none). Prints nothing when none match.
- `delete ID`: removes an expense and prints `deleted #ID`.
- `summary --month MONTH`: a line per category with expenses that month, `CATEGORY<TAB>TOTAL`, largest total first
  (equal totals by category name), then `total<TAB>TOTAL` for the month; just `total<TAB>0.00` if it has none. A
  category whose total is over its budget gets a third field, `over by X.XX`.
- `budget CATEGORY AMOUNT`: sets the category's monthly budget, replacing an earlier one, and prints
  `budget CATEGORY AMOUNT`. `budget CATEGORY 0` removes it and prints `budget CATEGORY removed`.
- `export FILE`: writes every expense to FILE as CSV with the header `id,date,amount,category,note`, by date then
  ID, amounts with two decimals, and prints `exported N expenses`.

Errors (an unknown or missing command, a missing or malformed argument, a bad amount, category, date or month, an
unknown ID): print a message starting with `error:` to stderr, exit with status 1, and leave `ledger.json` as it
was. Everything else exits with status 0.
