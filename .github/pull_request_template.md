## Summary

<!-- What does this PR do? Keep it to 1-2 sentences -->

## Type

- [ ] Bug fix
- [ ] New LLM interceptor
- [ ] New vector DB interceptor
- [ ] Attribution improvement
- [ ] Dashboard feature
- [ ] Performance optimization
- [ ] Documentation
- [ ] Tests only

## Changes

<!-- Bullet list of what changed -->

## Testing

- [ ] Unit tests added/updated: `pytest tests/ -m "not integration"` passes
- [ ] Integration tests pass (if applicable): `pytest tests/ -m integration`
- [ ] If dashboard changes: `cd dashboard && npm run build` succeeds
- [ ] Manual testing completed (describe below)

## Checklist

- [ ] Type hints on all new functions
- [ ] No `Any` types without justification
- [ ] If new interceptor: registered in `vectorlens/interceptors/__init__.py`
- [ ] If new interceptor: gracefully handles `ImportError` for optional dependencies
- [ ] No new required dependencies (optional deps are fine; add to `pyproject.toml`)
- [ ] Code formatted with `black` and `isort`
- [ ] CHANGELOG.md updated (add entry under `[Unreleased]`)
- [ ] Docstrings updated (if changing public API)

## Related Issues

<!-- Link to related issues: Closes #123 -->

## Notes for Reviewers

<!-- Any additional context or gotchas? -->
