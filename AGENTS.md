# RAS_DMFT.jl

Read `CONTRIBUTING.md` for the conventions.
What follows is procedure only.

## Commits

- Write a commit body for a future reader of `git log`,
  not for the reviewer of the current conversation.
  If the reasoning belongs in the pull request or the changelog, leave it out.

## Tests

- Iterate on the single test file that covers the change,
  as every test file loads its own dependencies and runs standalone.
  It runs about an order of magnitude faster than the full suite:

  ```sh
  cd test && julia --project=. -e 'include("natural_impurity_orbital.jl")'
  ```

  Run the full suite once before reporting the work done,
  either as `julia --project=. runtests.jl` from `test/`,
  or through the `Pkg.test` invocation of `CONTRIBUTING.md`,
  which precompiles the package as well.

- To show that a test fails on the unfixed version,
  take the source from the commit the branch started from,
  never from `HEAD` and never from `main`:

  ```sh
  cp src/FILE.jl /tmp/FILE.jl
  git show "$(git merge-base main HEAD)":src/FILE.jl > src/FILE.jl
  # run the test, then
  cp /tmp/FILE.jl src/FILE.jl && diff -q src/FILE.jl /tmp/FILE.jl
  ```

  `HEAD` is the unfixed state only while the fix is unstaged,
  and silently becomes the fixed state once it is committed,
  which makes the check pass for the wrong reason.
  `main` moves on as other work merges,
  so it can carry changes to the same file.

- Several test files assert `N * eps()` bounds on unseeded `rand` data.
  Set `N` by running the draw 1 000 000 times
  and rounding the biggest deviation up to a convenient value:
  a worst case of `128 * eps()` gives `N = 200`.
