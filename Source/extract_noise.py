import random
import json
import os

JUNK_APIS = [
  "GetSystemMetrics",
  "NtReadFile",
  "NtWriteFile",
  "NtCreateFile",
  "NtClose",
  "RegQueryValueExW",
  "timeGetTime",
  "RegOpenKeyExW",
  "RegCloseKey",
  "NtAllocateVirtualMemory",
  "NtQueryKey",
  "FindWindowA",
  "NtOpenFile",
  "NtDelayExecution",
  "LdrGetProcedureAddress",
  "NtQueryValueKey",
  "NtOpenKeyEx",
  "SetFilePointer",
  "DrawTextExW",
  "GetKeyState",
  "LdrLoadDll",
  "GetFileAttributesW",
  "GetSystemTimeAsFileTime",
  "NtQueryDirectoryFile",
  "NtFreeVirtualMemory",
  "CoInitializeEx",
  "GetCursorPos",
  "CoUninitialize",
  "GetForegroundWindow",
  "LdrGetDllHandle",
  "RegEnumKeyExW",
  "NtOpenKey",
  "SetErrorMode",
  "ReadProcessMemory",
  "CoCreateInstance",
  "NtDuplicateObject",
  "RegOpenKeyExA",
  "FindFirstFileExW",
  "LoadStringW",
  "GetFileSize",
  "NtMapViewOfSection",
  "__exception__",
  "NtQueryInformationFile",
  "NtUnmapViewOfSection",
  "FindWindowExW",
  "LoadResource",
  "RegQueryValueExA",
  "GetSystemWindowsDirectoryW",
  "SetFileTime",
  "NtProtectVirtualMemory",
  "CreateDirectoryW",
  "NtOpenProcess",
  "RegEnumKeyW",
  "SetEndOfFile",
  "NtCreateSection",
  "FindResourceExW",
  "CryptDecodeObjectEx",
  "RegCreateKeyExW",
  "NtQueryAttributesFile",
  "NtDeviceIoControlFile",
  "InternetReadFile",
  "SetFilePointerEx",
  "GetFileAttributesExW",
  "NtResumeThread",
  "FindResourceW",
  "SizeofResource",
  "RegSetValueExW",
  "GetAsyncKeyState",
  "GetSystemDirectoryW",
  "MoveFileWithProgressW",
  "LdrUnloadDll",
  "NtSetInformationFile",
  "SHGetFolderPathW",
  "RegEnumValueW",
  "LoadStringA",
  "CryptHashData",
  "RegEnumKeyExA",
  "EnumWindows",
  "DeviceIoControl",
  "RegSetValueExA",
  "GetSystemDirectoryA",
  "SetFileAttributesW",
  "CWindow_AddTimeoutCode",
  "Thread32Next",
  "NtEnumerateValueKey",
  "GetSystemWindowsDirectoryA",
  "PRF",
  "NtEnumerateKey",
  "RegCreateKeyExA",
  "DrawTextExA",
  "GetFileType",
  "CryptAcquireContextA",
  "GetFileInformationByHandleEx",
  "SearchPathW",
  "Process32NextW",
  "DeleteFileW",
  "GetFileSizeEx",
  "NtCreateMutant",
  "CreateToolhelp32Snapshot",
  "NtQuerySystemInformation"
]


_WIN_PATHS = [
    # File cấu hình / hệ thống cổ điển
    r"C:\Windows\win.ini",
    r"C:\Windows\System.ini",
    r"C:\Windows\System32\drivers\etc\hosts",

    # DLL / EXE hệ thống
    r"C:\Windows\System32\kernel32.dll",
    r"C:\Windows\System32\user32.dll",
    r"C:\Windows\System32\cmd.exe",
    r"C:\Windows\System32\notepad.exe",
    r"C:\Windows\System32\calc.exe",

    # Program Files (app phổ biến)
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Mozilla Firefox\firefox.exe",
    r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
    r"C:\Program Files\7-Zip\7z.exe",

    # ProgramData / AppData
    r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
    r"C:\ProgramData\chocolatey\logs\chocolatey.log",
    r"C:\Users\Public\AppData\Local\Temp\update.log",
    r"C:\Users\Public\Documents\readme.txt",
    r"C:\Users\Public\Downloads\setup.exe",

    # Temp / log / misc
    r"C:\Temp\log.txt",
    r"C:\Temp\debug.log",
    r"C:\Temp\cache.dat",
    r"C:\Users\Public\test.txt",
    r"C:\Users\Public\Desktop\shortcut.lnk",

    # Một số folder phổ biến khác
    r"C:\Windows\Temp\update.tmp",
    r"C:\Windows\Prefetch\NOTEPAD.EXE-12345678.pf",
    r"C:\Windows\Fonts\arial.ttf",
]


_REG_ROOTS = [
    # Nhánh Software chung
    r"Software\Example",
    r"Software\MyCompany\MyApp",
    r"Software\Microsoft\Windows\CurrentVersion",
    r"Software\Microsoft\Windows\CurrentVersion\Run",
    r"Software\Microsoft\Windows\CurrentVersion\RunOnce",
    r"Software\Microsoft\Windows\CurrentVersion\Explorer",
    r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall",

    # Policies / hệ thống
    r"Software\Policies\Microsoft\Windows\System",
    r"Software\Policies\Microsoft\Windows\Explorer",
    r"Software\Policies\Microsoft\Windows\WindowsUpdate",

    # Services / driver
    r"System\CurrentControlSet\Services",
    r"System\CurrentControlSet\Services\EventLog",
    r"System\CurrentControlSet\Services\Tcpip",
    r"System\CurrentControlSet\Services\LanmanWorkstation",

    # Control / Session Manager / ControlSet
    r"System\CurrentControlSet\Control\Session Manager",
    r"System\CurrentControlSet\Control\Terminal Server",
    r"System\CurrentControlSet\Control\Lsa",

    # User-specific (thường dùng với HKEY_CURRENT_USER)
    r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
    r"Software\Microsoft\Windows\CurrentVersion\Explorer\RecentDocs",
    r"Software\Microsoft\Windows\CurrentVersion\Explorer\RunMRU",
]


_DLL_NAMES = [
    # Core Windows DLLs
    "kernel32.dll",
    "user32.dll",
    "gdi32.dll",
    "advapi32.dll",
    "ntdll.dll",
    "shell32.dll",
    "ole32.dll",
    "oleaut32.dll",
    "comdlg32.dll",
    "comctl32.dll",
    "shlwapi.dll",
    "rpcrt4.dll",

    # Network / Internet
    "wininet.dll",
    "ws2_32.dll",
    "wsock32.dll",
    "urlmon.dll",
    "secur32.dll",

    # Crypto / security
    "crypt32.dll",
    "cryptui.dll",

    # C runtime / common libs
    "msvcrt.dll",
    "ucrtbase.dll",
    "vcruntime140.dll",

    # Printing / spooler
    "winspool.drv",

    # Misc UI / multimedia
    "uxtheme.dll",
    "dwmapi.dll",
    "version.dll",
    "dbghelp.dll",
]


def _rand_hex_dword(min_val=0x1000, max_val=0xFFFFF):
    return "0x%08X" % random.randint(min_val, max_val)

def build_junk_args(api_name):
    name = api_name
    lower = api_name.lower()

    # ===== System metrics / time / system dirs =====
    if name == "GetSystemMetrics":
        return {"nIndex": str(random.choice([0, 1, 16, 80]))}

    if name == "timeGetTime":
        return {}

    if name == "GetSystemTimeAsFileTime":
        return {"lpSystemTimeAsFileTime": _rand_hex_dword()}

    if name in ("GetSystemWindowsDirectoryW", "GetSystemWindowsDirectoryA"):
        return {
            "lpBuffer": random.choice(_WIN_PATHS),
            "uSize": "260",
        }

    if name in ("GetSystemDirectoryW", "GetSystemDirectoryA"):
        return {
            "lpBuffer": random.choice(_WIN_PATHS),
            "uSize": "260",
        }

    if name == "SHGetFolderPathW":
        return {
            "hwndOwner": _rand_hex_dword(),
            "nFolder": "CSIDL_APPDATA",
            "pszPath": r"C:\Users\Public\AppData",
        }

    if name == "NtQuerySystemInformation":
        return {
            "SystemInformationClass": random.choice(["SystemProcessInformation", "SystemModuleInformation"]),
            "SystemInformationLength": str(random.choice([1024, 4096])),
        }

    if name == "SetErrorMode":
        return {"uMode": "0x0001"}

    # ===== File I/O & NT file =====
    if name in ("NtReadFile", "NtWriteFile"):
        return {
            "FileHandle": _rand_hex_dword(),
            "ByteOffset": _rand_hex_dword(),
            "Length": str(random.choice([256, 512, 1024, 4096])),
            "Key": _rand_hex_dword(),
        }

    if name == "NtCreateFile":
        return {
            "FileHandle": _rand_hex_dword(),
            "FileName": random.choice(_WIN_PATHS),
            "DesiredAccess": "0x120089",
            "ShareAccess": "0x3",
            "CreateDisposition": "OPEN_EXISTING",
        }

    if name in ("NtClose",):
        return {"Handle": _rand_hex_dword()}

    if name in ("NtOpenFile",):
        return {
            "FileHandle": _rand_hex_dword(),
            "FileName": random.choice(_WIN_PATHS),
            "DesiredAccess": "0x120089",
            "ShareAccess": "0x3",
        }

    if name in ("SetFilePointer", "SetFilePointerEx"):
        return {
            "hFile": _rand_hex_dword(),
            "lDistanceToMove": str(random.choice([0, 128, 1024])),
            "dwMoveMethod": random.choice(["FILE_BEGIN", "FILE_END", "FILE_CURRENT"]),
        }

    if name in ("GetFileSize", "GetFileSizeEx"):
        return {
            "hFile": _rand_hex_dword(),
            "lpFileSizeHigh": _rand_hex_dword(),
        }

    if "getfileattributes" in lower:
        return {"lpFileName": random.choice(_WIN_PATHS)}

    if name == "GetFileAttributesExW":
        return {
            "lpFileName": random.choice(_WIN_PATHS),
            "fInfoLevelId": "GetFileExInfoStandard",
            "lpFileInformation": _rand_hex_dword(),
        }

    if name == "DeleteFileW":
        return {"lpFileName": random.choice(_WIN_PATHS)}

    if name == "CreateDirectoryW":
        return {"lpPathName": r"C:\Temp\JunkDir", "lpSecurityAttributes": _rand_hex_dword()}

    if name == "SetFileTime":
        return {
            "hFile": _rand_hex_dword(),
            "lpCreationTime": _rand_hex_dword(),
            "lpLastAccessTime": _rand_hex_dword(),
            "lpLastWriteTime": _rand_hex_dword(),
        }

    if name == "SetEndOfFile":
        return {"hFile": _rand_hex_dword()}

    if name in ("GetFileType",):
        return {"hFile": _rand_hex_dword()}

    if name == "GetFileInformationByHandleEx":
        return {
            "hFile": _rand_hex_dword(),
            "FileInformationClass": "FileBasicInfo",
        }

    if name in ("NtQueryInformationFile", "NtQueryAttributesFile"):
        return {
            "FileHandle": _rand_hex_dword(),
            "FileInformationClass": "FileBasicInformation",
        }

    if name == "NtQueryDirectoryFile":
        return {
            "FileHandle": _rand_hex_dword(),
            "FileName": random.choice(_WIN_PATHS),
            "FileInformationClass": "FileDirectoryInformation",
        }

    if name in ("NtSetInformationFile",):
        return {
            "FileHandle": _rand_hex_dword(),
            "FileInformationClass": "FileBasicInformation",
        }

    if name == "MoveFileWithProgressW":
        return {
            "lpExistingFileName": random.choice(_WIN_PATHS),
            "lpNewFileName": random.choice(_WIN_PATHS),
        }

    # ===== Memory / section / process =====
    if name in ("NtAllocateVirtualMemory", "NtFreeVirtualMemory", "NtProtectVirtualMemory"):
        return {
            "ProcessHandle": _rand_hex_dword(),
            "BaseAddress": _rand_hex_dword(),
            "RegionSize": str(random.choice([4096, 8192, 16384])),
            "AllocationType": "0x1000",
            "Protect": "PAGE_READWRITE",
        }

    if name in ("NtMapViewOfSection", "NtUnmapViewOfSection"):
        return {
            "SectionHandle": _rand_hex_dword(),
            "ProcessHandle": _rand_hex_dword(),
            "BaseAddress": _rand_hex_dword(),
        }

    if name == "NtCreateSection":
        return {
            "SectionHandle": _rand_hex_dword(),
            "DesiredAccess": "0xF001F",
            "MaximumSize": str(random.choice([4096, 65536])),
        }

    if name == "NtOpenProcess":
        return {
            "ProcessHandle": _rand_hex_dword(),
            "DesiredAccess": "0x1FFFFF",
            "ProcessId": str(random.randint(100, 5000)),
        }

    if name == "NtDuplicateObject":
        return {
            "SourceProcessHandle": _rand_hex_dword(),
            "SourceHandle": _rand_hex_dword(),
            "TargetProcessHandle": _rand_hex_dword(),
        }

    if name == "NtResumeThread":
        return {
            "ThreadHandle": _rand_hex_dword(),
            "SuspendCount": "0",
        }

    if name == "NtCreateMutant":
        return {"MutantHandle": _rand_hex_dword(), "DesiredAccess": "0x1F0001"}

    if name == "ReadProcessMemory":
        return {
            "hProcess": _rand_hex_dword(),
            "lpBaseAddress": _rand_hex_dword(),
            "nSize": str(random.choice([256, 512, 1024])),
        }

    if name == "CreateToolhelp32Snapshot":
        return {
            "dwFlags": "TH32CS_SNAPPROCESS",
            "th32ProcessID": "0",
        }

    if name in ("Thread32Next", "Process32NextW"):
        return {
            "hSnapshot": _rand_hex_dword(),
            "lppe": _rand_hex_dword(),
        }

    # ===== Registry =====
    if name in ("RegOpenKeyExW", "RegOpenKeyExA"):
        return {
            "hKey": random.choice(["HKEY_CURRENT_USER", "HKEY_LOCAL_MACHINE"]),
            "lpSubKey": random.choice(_REG_ROOTS),
            "ulOptions": "0x0",
            "samDesired": "0x20019",
        }

    if name in ("RegCreateKeyExW", "RegCreateKeyExA"):
        return {
            "hKey": random.choice(["HKEY_CURRENT_USER", "HKEY_LOCAL_MACHINE"]),
            "lpSubKey": random.choice(_REG_ROOTS) + r"\Junk",
            "dwOptions": "0x0",
            "samDesired": "0xF003F",
        }

    if name == "RegCloseKey":
        return {"hKey": _rand_hex_dword()}

    if name in ("RegQueryValueExW", "RegQueryValueExA"):
        return {
            "hKey": _rand_hex_dword(),
            "lpValueName": random.choice(["Path", "InstallLocation", "Version"]),
            "lpType": "REG_SZ",
        }

    if name in ("RegSetValueExW", "RegSetValueExA"):
        return {
            "hKey": _rand_hex_dword(),
            "lpValueName": random.choice(["JunkValue", "TempDir"]),
            "dwType": "REG_SZ",
            "lpData": random.choice([r"C:\Temp", r"C:\Users\Public"]),
        }

    if name in ("RegEnumKeyExW", "RegEnumKeyExA", "RegEnumKeyW"):
        return {
            "hKey": _rand_hex_dword(),
            "dwIndex": str(random.randint(0, 10)),
            "lpName": "SubKey",
        }

    if name == "RegEnumValueW":
        return {
            "hKey": _rand_hex_dword(),
            "dwIndex": str(random.randint(0, 20)),
            "lpValueName": "ValueName",
        }

    if name in ("NtOpenKey", "NtOpenKeyEx"):
        return {
            "KeyHandle": _rand_hex_dword(),
            "KeyName": random.choice(_REG_ROOTS),
        }

    if name == "NtQueryKey":
        return {
            "KeyHandle": _rand_hex_dword(),
            "KeyInformationClass": "KeyBasicInformation",
        }

    if name == "NtQueryValueKey":
        return {
            "KeyHandle": _rand_hex_dword(),
            "ValueName": random.choice(["Path", "Config", "Version"]),
        }

    if name == "NtEnumerateValueKey":
        return {
            "KeyHandle": _rand_hex_dword(),
            "Index": str(random.randint(0, 20)),
        }

    if name == "NtEnumerateKey":
        return {
            "KeyHandle": _rand_hex_dword(),
            "Index": str(random.randint(0, 10)),
        }

    # ===== Window / UI =====
    if name == "FindWindowA":
        return {
            "lpClassName": random.choice(["Notepad", "Chrome_WidgetWin_1", None]),
            "lpWindowName": random.choice(["Untitled - Notepad", "MyApp"]),
        }

    if name == "FindWindowExW":
        return {
            "hwndParent": _rand_hex_dword(),
            "hwndChildAfter": _rand_hex_dword(),
            "lpszClass": "Edit",
            "lpszWindow": None,
        }

    if name == "GetForegroundWindow":
        return {}

    if name == "GetCursorPos":
        return {"lpPoint": _rand_hex_dword()}

    if name in ("GetKeyState", "GetAsyncKeyState"):
        return {"nVirtKey": "0x41"}  # 'A'

    if name in ("DrawTextExW", "DrawTextExA"):
        return {
            "hdc": _rand_hex_dword(),
            "lpchText": "Junk Text",
            "cchText": "9",
        }

    if name == "EnumWindows":
        return {"lpEnumFunc": _rand_hex_dword(), "lParam": _rand_hex_dword()}

    if name == "CWindow_AddTimeoutCode":
        return {"timeout": str(random.choice([100, 500, 1000]))}

    # ===== Loader / resource / strings =====
    if name in ("LdrLoadDll",):
        return {
            "DllName": random.choice(_DLL_NAMES),
        }

    if name in ("LdrGetDllHandle",):
        return {
            "DllName": random.choice(_DLL_NAMES),
        }

    if name == "LdrGetProcedureAddress":
        return {
            "ModuleHandle": _rand_hex_dword(),
            "FunctionName": random.choice(["CreateFileW", "WriteFile", "ReadFile"]),
        }

    if name in ("LoadStringW", "LoadStringA"):
        return {
            "hInstance": _rand_hex_dword(),
            "uID": str(random.randint(1, 500)),
            "lpBuffer": _rand_hex_dword(),
            "cchBufferMax": str(random.choice([128, 256])),
        }

    if name == "LoadResource":
        return {
            "hModule": _rand_hex_dword(),
            "hResInfo": _rand_hex_dword(),
        }

    if name in ("FindResourceExW", "FindResourceW"):
        return {
            "hModule": _rand_hex_dword(),
            "lpName": str(random.randint(1, 500)),
            "lpType": "RT_RCDATA",
        }

    if name == "SizeofResource":
        return {
            "hModule": _rand_hex_dword(),
            "hResInfo": _rand_hex_dword(),
        }

    if name == "LdrUnloadDll":
        return {"BaseAddress": _rand_hex_dword()}

    if name == "SearchPathW":
        return {
            "lpPath": r"C:\Windows\System32",
            "lpFileName": random.choice(_DLL_NAMES),
            "lpExtension": ".dll",
            "lpBuffer": _rand_hex_dword(),
        }

    # ===== COM =====
    if name in ("CoInitializeEx", "CoUninitialize"):
        return {}

    if name == "CoCreateInstance":
        return {
            "rclsid": "{00000000-0000-0000-C000-000000000046}",
            "dwClsContext": "CLSCTX_INPROC_SERVER",
        }

    # ===== Crypto =====
    if name == "CryptDecodeObjectEx":
        return {
            "lpszStructType": "X509_BASIC_CONSTRAINTS",
            "pbEncoded": _rand_hex_dword(),
        }

    if name == "CryptHashData":
        return {
            "hHash": _rand_hex_dword(),
            "dwDataLen": str(random.choice([16, 32, 64])),
        }

    if name == "CryptAcquireContextA":
        return {
            "pszContainer": None,
            "pszProvider": "Microsoft Enhanced Cryptographic Provider v1.0",
            "dwProvType": "PROV_RSA_FULL",
        }

    # ===== Network / Device IO =====
    if name in ("DeviceIoControl", "NtDeviceIoControlFile"):
        return {
            "FileHandle": _rand_hex_dword(),
            "IoControlCode": "0x00070000",
        }

    if name == "InternetReadFile":
        return {
            "hFile": _rand_hex_dword(),
            "dwNumberOfBytesToRead": str(random.choice([512, 1024, 4096])),
        }

    # ===== Misc special =====
    if name == "__exception__":
        return {"Code": "0xC0000005"}

    if name == "NtDelayExecution":
        return {
            "Alertable": "FALSE",
            "DelayInterval": _rand_hex_dword(),
        }

    if name == "PRF":
        return {}

    # Mặc định: không cần arg → để rỗng
    return {}


def inject_junk_apis(features, noise_ratio=0.1):
    """
    Thêm junk API vào features["api_call_sequence"].

    - noise_ratio = 0.1 → số call rác ≈ 10% số call gốc.
    - Chỉ random một số lượng junk cố định, không cố gắng dùng hết JUNK_APIS.
    """
    seq = features.get("api_call_sequence", [])
    n = len(seq)
    if n == 0 or noise_ratio <= 0:
        return features

    # Số lượng junk call muốn chèn
    n_junk = max(1, int(n * noise_ratio))

    # Khoảng cách trung bình giữa các junk
    step_c = max(1, n // n_junk)

    new_seq = []
    junk_inserted = 0

    for i, call in enumerate(seq):
        new_seq.append(call)

        # Nếu chưa đủ số junk, và đủ “khoảng cách” thì chèn thêm 1 junk
        if junk_inserted < n_junk and (i % step_c == 0):
            base = call  # dùng call hiện tại làm template (pid, category, ...)
            api_name = random.choice(JUNK_APIS)
            base_time = base.get("time", "")

            junk_call = {
                "time": base_time,
                "api": api_name,
                "pid": base.get("pid"),
                "arguments": build_junk_args(api_name),
                "category": base.get("category", "")
            }

            new_seq.append(junk_call)
            print(" Inject junk call:", junk_call)
            print()
            junk_inserted += 1


    features["api_call_sequence"] = new_seq
    return features



def extract_list(d, key):
    return d.get(key, []) if isinstance(d.get(key), list) else []

def extract_features(report):
    features = {}

    # 1) Behavior summary
    summary = report.get("behavior", {}).get("summary", {})

    features["behavior_summary"] = summary

    # 2) Dropped files
    features["dropped_files"] = report.get("dropped", [])

    # 3) API statistics
    # features["api_statistics"] = report.get("behavior", {}).get("apistats", {})

    # 4) API call sequence
    api_calls = []
    for proc in report.get("behavior", {}).get("processes", []):
        for call in proc.get("calls", []):
            api_calls.append({
                "time": call.get("time"),
                "api": call.get("api"),
                "pid": proc.get("pid"),
                "arguments": call.get("arguments", {}),
                "category": call.get("category", "")
            })
    api_calls.sort(key=lambda x: x["time"])
    features["api_call_sequence"] = api_calls

    # 5) Network activity
    net = report.get("network", {})
    features["network"] = {
        "http": net.get("http", []),
        "tcp": net.get("tcp", []),
        "udp": net.get("udp", []),
        "dns": net.get("dns", []),
        "hosts": net.get("hosts", []),
        "icmp": net.get("icmp", []),
        "smtp": net.get("smtp", [])
    }

    # 6) Static imports
    static = report.get("static", {})
    imports = []
    for dll in static.get("pe_imports", []):
        for imp in dll.get("imports", []):
            name = imp.get("name")
            if name:
                imports.append(name)
    features["static_imports"] = imports

    # 7) Unified Signatures (gộp cả TTP & full)
    features["signatures"] = []
    for sig in report.get("signatures", []):
        features["signatures"].append({
            "node_type":   "Signature",
            "name":        sig.get("name", ""),
            "description": sig.get("description", ""),
            "severity":    sig.get("severity", 0),
            "ttp_ids":     list(sig.get("ttp", {}).keys()),
            "markcount":   sig.get("markcount", 0),
            "marks":       sig.get("marks", [])
        })


    # 9) Process details
    processes = []
    for p in report.get("behavior", {}).get("processes", []):
        processes.append({
            "name": p.get("process_name"),
            "path": p.get("process_path"),
            "pid": p.get("pid"),
            "cmdline": p.get("command_line"),
            "modules": [
                m.get("basename")
                for m in p.get("modules", [])
                if m.get("basename")
            ]
        })
    features["processes"] = processes

    return features

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                with open(input_path, "r", encoding="utf-8", errors="replace") as f:
                    report = json.load(f)

                # Kiểm tra nếu KHÔNG có bất kỳ calls nào trong processes
                has_calls = False
                for proc in report.get("behavior", {}).get("processes", []):
                    if proc.get("calls"):
                        has_calls = True
                        break

                # if not has_calls:
                #    print(f"⏭️  Bỏ qua {filename} (không có API calls)")
                #    continue

                features = extract_features(report)
                features = inject_junk_apis(features, noise_ratio=0.10)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(features, f, indent=2, ensure_ascii=True)
                
                with open("log_extractdata.txt", "a", encoding="utf-8") as log_file:
                    log_file.write(f"Đã xử lý: {filename}\n")
                print(f"✅ Đã xử lý: {filename}")
            except Exception as e:
                with open("log_extractdata.txt", "a", encoding="utf-8") as log_file:
                    log_file.write(f"Lỗi với: {filename}\n")
                print(f"❌ Lỗi với {filename}: {e}")


def main():
    process_folder("ori_data", "data_ransomware_extracted_noise/ransomware")
    #process_folder("/home/xlstm/work/cuckoo_benign_reports", "reports/benign")

if __name__ == "__main__":
    main()