$installPath = &"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -property installationpath
Import-Module (Join-Path $installPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
Enter-VsDevShell -VsInstallPath $installPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -no_logo'