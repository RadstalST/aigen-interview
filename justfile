init:
  pixi install
  pre-commit install
  pre-commit install --hook-type commit-msg
  pre-commit install --hook-type pre-push
  pre-commit install --hook-type pre-commit

lint:
  pre-commit run --all-files

commit:
  just lint
  cz commit