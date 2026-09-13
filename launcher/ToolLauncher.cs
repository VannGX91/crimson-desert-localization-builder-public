using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class ToolLauncher
{
    private const string ProductName = "Crimson Desert Localization Builder";

    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            return Run(root, args);
        }
        catch (Exception error)
        {
            if (args.Length == 1 && args[0] == "--smoke-test") return 1;
            return Fail("Không thể khởi động công cụ.\n\n" + error.Message);
        }
    }

    private static int Run(string root, string[] args)
    {
        string python = Path.Combine(root, "_internal", "runtime", "pythonw.exe");
        string application = Path.Combine(root, "_internal", "app", "app.py");

        if (!File.Exists(python) || !File.Exists(application))
        {
            return Fail(
                "Thiếu dữ liệu chương trình trong thư mục _internal.\n\n" +
                "Vui lòng giải nén lại toàn bộ file ZIP và giữ file EXE " +
                "cùng thư mục _internal."
            );
        }
        if (args.Length > 1 || (args.Length == 1 && args[0] != "--smoke-test"))
        {
            return Fail("Tham số khởi động không được hỗ trợ.");
        }

        string childArguments = "-E -s -X utf8 -B " + Quote(application);
        if (args.Length == 1)
        {
            childArguments += " --smoke-test";
        }

        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = python,
            Arguments = childArguments,
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        startInfo.EnvironmentVariables["CRIMSON_LAUNCHER_DIRECTORY"] = root;

        using (Process child = Process.Start(startInfo))
        {
            if (child == null)
            {
                return Fail("Không thể khởi động tiến trình ứng dụng.");
            }
            child.WaitForExit();
            if (child.ExitCode != 0 && args.Length == 0)
            {
                return Fail(
                    "Ứng dụng đã dừng trong lúc khởi động.\n\n" +
                    "Hãy giải nén lại đầy đủ folder của tool và bảo đảm " +
                    "Microsoft Edge WebView2 Runtime đã được cài đặt.\n\n" +
                    "Mã lỗi: " + child.ExitCode
                );
            }
            return child.ExitCode;
        }
    }

    private static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }

    private static int Fail(string message)
    {
        MessageBox.Show(message, ProductName, MessageBoxButtons.OK, MessageBoxIcon.Error);
        return 1;
    }
}
