import Foundation

let fileManager = FileManager.default
let documentsDirectory = fileManager.homeDirectoryForCurrentUser
    .appendingPathComponent("Documents", isDirectory: true)

func isValidLauncher(_ url: URL) -> Bool {
    let projectDirectory = url.deletingLastPathComponent()
    let python = projectDirectory.appendingPathComponent(".venv/bin/python")
    let helper = projectDirectory.appendingPathComponent(
        "launcher-assets/launch_webui.py"
    )
    let model = projectDirectory.appendingPathComponent(
        "models/mlx-IndexTTS-2.5-int8",
        isDirectory: true
    )
    return fileManager.isExecutableFile(atPath: url.path)
        && fileManager.isExecutableFile(atPath: python.path)
        && fileManager.fileExists(atPath: helper.path)
        && fileManager.fileExists(atPath: model.path)
}

func findLauncher() -> URL? {
    let directLauncher = documentsDirectory
        .appendingPathComponent("创意项目/mlx-indextts", isDirectory: true)
        .appendingPathComponent("启动IndexTTS2.command")
    if isValidLauncher(directLauncher) {
        return directLauncher
    }

    guard let documentFolders = try? fileManager.contentsOfDirectory(
        at: documentsDirectory,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles]
    ) else {
        return nil
    }
    for folder in documentFolders {
        let launcher = folder
            .appendingPathComponent("创意项目/mlx-indextts", isDirectory: true)
            .appendingPathComponent("启动IndexTTS2.command")
        if isValidLauncher(launcher) {
            return launcher
        }
    }
    return nil
}

guard let launcher = findLauncher() else {
    let alert = Process()
    alert.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
    alert.arguments = [
        "-e",
        "display alert \"找不到 IndexTTS2\" message \"已在‘文稿’目录中搜索，但没有找到完整的 IndexTTS2 项目。\" as critical"
    ]
    try? alert.run()
    alert.waitUntilExit()
    exit(1)
}

// Terminal already has the user's normal Documents-folder access. Launching
// the command there avoids macOS privacy controls suspending a background-only
// app while it tries to enter the project directory.
let task = Process()
task.executableURL = URL(fileURLWithPath: "/usr/bin/open")
task.arguments = ["-a", "Terminal", launcher.path]

do {
    try task.run()
    task.waitUntilExit()
    exit(task.terminationStatus)
} catch {
    let alert = Process()
    alert.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
    alert.arguments = [
        "-e",
        "display alert \"IndexTTS2 启动失败\" message \"无法打开启动命令。\" as critical"
    ]
    try? alert.run()
    alert.waitUntilExit()
    exit(1)
}
