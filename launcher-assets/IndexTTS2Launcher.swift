import Foundation

let projectDirectory = Bundle.main.bundleURL
    .deletingLastPathComponent()
    .deletingLastPathComponent()
let launcher = projectDirectory
    .appendingPathComponent("启动IndexTTS2.command")
    .path

guard FileManager.default.isExecutableFile(atPath: launcher) else {
    let alert = Process()
    alert.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
    alert.arguments = ["-e", "display alert \"找不到 IndexTTS2\" message \"启动脚本不存在或没有执行权限。\" as critical"]
    try? alert.run()
    alert.waitUntilExit()
    exit(1)
}

// Terminal already has the user's normal Documents-folder access. Launching
// the command there avoids macOS privacy controls suspending a background-only
// app while it tries to enter the project directory.
let task = Process()
task.executableURL = URL(fileURLWithPath: "/usr/bin/open")
task.arguments = ["-a", "Terminal", launcher]

do {
    try task.run()
    task.waitUntilExit()
    exit(task.terminationStatus)
} catch {
    let alert = Process()
    alert.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
    alert.arguments = ["-e", "display alert \"IndexTTS2 启动失败\" message \"无法打开启动命令。\" as critical"]
    try? alert.run()
    alert.waitUntilExit()
    exit(1)
}
