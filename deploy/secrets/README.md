# Local secret files

The `*.example` files in this directory contain development-only placeholders. Copy
each required file to the same name without `.example`, replace its value with a
unique secret, and point Compose at it through the matching `*_SECRET_FILE`
variable. Files without `.example` are ignored by Git.

Never reuse these placeholder values in a shared or production environment.
