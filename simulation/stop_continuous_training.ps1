$trainingProcesses = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*train_continuous.py*' -or $_.CommandLine -like '*train_stabilized_cpu.py*' -or $_.CommandLine -like '*train_warp_continuous.py*' }
foreach ($trainingProcess in $trainingProcesses) { Stop-Process -Id $trainingProcess.ProcessId -ErrorAction SilentlyContinue }
Write-Output 'Stopped continuous training. The most recently completed iteration remains saved.'
