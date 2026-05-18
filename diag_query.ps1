$tcpClient = New-Object System.Net.Sockets.TcpClient('localhost', 19845)
$stream = $tcpClient.GetStream()
$writer = New-Object System.IO.StreamWriter($stream)
$reader = New-Object System.IO.StreamReader($stream)
$writer.WriteLine('{"cmd":"overlayDiag"}')
$writer.Flush()
Start-Sleep -Milliseconds 500
$reader.ReadLine()
$tcpClient.Close()
