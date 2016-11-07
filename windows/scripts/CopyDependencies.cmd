set DLL_Path=%~1%
set OUT_DIR=%~2%
set DLLs=%~3%

echo copy "%DLLs%" dlls to "%OUT_DIR%"
robocopy /NS /NC /NFL /NDL /NP /NJH /NJS "%DLL_Path%" %OUT_DIR% %DLLs%
if errorlevel 1 (exit 0)
