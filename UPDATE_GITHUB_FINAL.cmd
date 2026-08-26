@echo off
setlocal
cd /d E:\OutSourcing\running_ai_prj\RunningAI-Core
if errorlevel 1 goto :error

echo ============================================================
echo RunningAI final model deployment

echo ============================================================
echo [1/6] Current repository status
git status --short

echo.
echo [2/6] Final model artifact verification
python api\scripts\verify_final_api_model.py
if errorlevel 1 goto :error

echo.
echo [3/6] API test suite
pushd api
python -m pytest -q
if errorlevel 1 (
  popd
  goto :error
)
popd

echo.
echo [4/6] Integration QA
python scripts\validate_integration.py
if errorlevel 1 goto :error

echo.
echo [5/6] Commit final deployment
git add -A
git diff --cached --quiet
if not errorlevel 1 goto :no_changes

git commit -m "Deploy final validated model and API v1.4.0"
if errorlevel 1 goto :error

echo.
echo [6/6] Push main
git push origin main
if errorlevel 1 goto :error

goto :done

:no_changes
echo No staged changes. Nothing to commit.
echo Checking remote push state...
git push origin main
if errorlevel 1 goto :error

goto :done

:done
echo.
echo ============================================================
echo Update completed successfully.
echo ============================================================
git status --short
git log -1 --oneline
exit /b 0

:error
echo.
echo ============================================================
echo Update failed. Do not force-push. Review the error above.
echo ============================================================
git status --short
exit /b 1
