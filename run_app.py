import os
import sys
import subprocess


def main():
    # 打包后获取临时目录
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    ap8_path = os.path.join(base_dir, "ap8.py")
    print(f"✅ 找到主程序文件：{ap8_path}")
    print(f"✅ 当前工作目录：{os.getcwd()}")
    print(f"✅ data文件夹路径：{os.path.join(base_dir, 'data')}")
    print("=" * 50)
    print("正在启动Streamlit服务...")
    print("如果服务启动失败，下面会直接显示错误原因！")
    print("=" * 50)

    # 关键：把stdout和stderr直接输出到控制台，显示所有错误
    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                ap8_path,
                "--server.port=8501",
                "--server.headless=true",
                "--browser.gatherUsageStats=false"
            ],
            check=True,
            stdout=sys.stdout,  # 把Streamlit的标准输出直接打印到控制台
            stderr=sys.stderr  # 把Streamlit的错误输出也直接打印到控制台
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit启动失败，错误码：{e.returncode}")
    except Exception as e:
        print(f"❌ 未知错误：{str(e)}")


if __name__ == "__main__":
    main()