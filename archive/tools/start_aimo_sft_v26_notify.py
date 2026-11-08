"""Local VS Code notification extension receives completion signals from Colab."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def start_terminal_event(title):
    # Google Colab 0.9.7 uses a custom PTY without injected shell integration.
    # Advertise one command using VS Code's documented OSC 633 protocol so the
    # local extension can subscribe to its output stream, including OSC 777.
    if os.environ.get('VSCODE_SHELL_INTEGRATION')=='1':return False
    clean=''.join(ch for ch in title if ch.isalnum() or ch in ' -_')
    print('\033]633;A\007\033]633;B\007\033]633;E;'+clean+'\007\033]633;C\007',end='',flush=True)
    time.sleep(.3)
    return True


def end_terminal_event(synthetic,code):
    if synthetic:print(f'\033]633;D;{code}\007',end='',flush=True)


def notify(title,message):
    def clean(value):
        return ''.join(ch for ch in value if ch>=' ' and ch not in '\x7f;').strip()
    print('\033]777;notify;'+clean(title)+';'+clean(message)+'\007',end='',flush=True)
    print('\n[任务通知] '+title+'：'+message,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-notification',action='store_true')
    p.add_argument('--title',default='v26 数学 SFT')
    p.add_argument('--command',nargs=argparse.REMAINDER,help='Optional other command to run and notify on completion')
    a,training_args=p.parse_known_args()
    synthetic=start_terminal_event(a.title)
    if a.test_notification:
        notify('Colab 通知测试','如果你看到右下角弹窗，任务结束提醒已接通。')
        end_terminal_event(synthetic,0)
        return 0
    if a.command:
        if training_args:p.error('Use --command as the last option')
        command=a.command
    else:
        starter=Path(__file__).with_name('start_aimo_sft_v26.py')
        if not starter.exists():raise FileNotFoundError('Missing sibling file: '+str(starter))
        command=[sys.executable,str(starter),*training_args]
    started=time.monotonic()
    try:
        result=subprocess.run(command)
        minutes=(time.monotonic()-started)/60
        if result.returncode==0:
            notify(a.title+' 已完成',f'命令成功结束，耗时 {minutes:.1f} 分钟。请查看终端和输出文件。')
        else:
            notify(a.title+' 运行失败',f'退出码 {result.returncode}，耗时 {minutes:.1f} 分钟。请返回终端查看错误。')
        end_terminal_event(synthetic,result.returncode)
        return result.returncode
    except KeyboardInterrupt:
        notify(a.title+' 已中断','任务被手动中断，请查看最近的 checkpoint。')
        end_terminal_event(synthetic,130)
        return 130
    except OSError as error:
        notify(a.title+' 启动失败',str(error))
        end_terminal_event(synthetic,1)
        raise


if __name__=='__main__':sys.exit(main())
