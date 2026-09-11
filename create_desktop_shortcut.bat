@echo off
set "VBS_SCRIPT=%TEMP%\make_shortcut.vbs"
set "ICON_PATH=%~dp0satquery_icon.ico"

echo Set oWS = WScript.CreateObject("WScript.Shell") > "%VBS_SCRIPT%"

echo sLink1 = "%USERPROFILE%\OneDrive\Desktop\Start SatQuery AI.lnk" >> "%VBS_SCRIPT%"
echo Set oLink1 = oWS.CreateShortcut(sLink1) >> "%VBS_SCRIPT%"
echo oLink1.TargetPath = "%~dp0start_app.bat" >> "%VBS_SCRIPT%"
echo oLink1.WorkingDirectory = "%~dp0" >> "%VBS_SCRIPT%"
echo oLink1.IconLocation = "%ICON_PATH%" >> "%VBS_SCRIPT%"
echo oLink1.Save >> "%VBS_SCRIPT%"

echo sLink2 = "%USERPROFILE%\Desktop\Start SatQuery AI.lnk" >> "%VBS_SCRIPT%"
echo Set oLink2 = oWS.CreateShortcut(sLink2) >> "%VBS_SCRIPT%"
echo oLink2.TargetPath = "%~dp0start_app.bat" >> "%VBS_SCRIPT%"
echo oLink2.WorkingDirectory = "%~dp0" >> "%VBS_SCRIPT%"
echo oLink2.IconLocation = "%ICON_PATH%" >> "%VBS_SCRIPT%"
echo oLink2.Save >> "%VBS_SCRIPT%"

cscript /nologo "%VBS_SCRIPT%"
if exist "%VBS_SCRIPT%" del "%VBS_SCRIPT%"

echo Desktop shortcuts updated with custom SatQuery AI icon!
