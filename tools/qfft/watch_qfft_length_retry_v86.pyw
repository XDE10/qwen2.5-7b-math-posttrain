"""Read-only monitor. Closing the window never stops the two-worker task."""
from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk

OUT = Path(__file__).resolve().parents[1] / 'artifacts/qfft_length_retry_v86'
root = tk.Tk()
root.title('项目一 | 41题 max→high 有界重跑 v86')
root.geometry('840x520')
tk.Label(root, text='41题有界重跑 · 2题并发', font=('Microsoft YaHei', 19, 'bold')).pack(pady=15)
tk.Label(root, text='47题原high已跳过；原1000份不覆盖。关闭本窗口不会停止后台任务。').pack()
banner = tk.Label(root, font=('Microsoft YaHei', 12), wraplength=790, pady=12)
banner.pack(fill='x', padx=15, pady=10)
bar = ttk.Progressbar(root, maximum=41, length=780)
bar.pack(pady=5)
detail = tk.Label(root, justify='left', anchor='w', font=('Microsoft YaHei', 11))
detail.pack(fill='x', padx=20, pady=10)
log = tk.Text(root, height=12, bg='#111827', fg='white', wrap='word')
log.pack(fill='both', expand=True, padx=15, pady=10)


def refresh():
    try:
        p = OUT / 'status.json'
        s = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        state = s.get('state', 'starting')
        colour = '#fee2e2' if state == 'paused_requires_review' else '#dcfce7'
        banner.config(text=f'状态：{state} | 完成 {s.get("completed", 0)}/41', bg=colour)
        bar['value'] = s.get('completed', 0)
        detail.config(text=f'结构完整待审：{s.get("complete_pending_review",0)}    两次失败跳过：{s.get("both_failed_skip",0)}\n'
                      f'当前题目/档位：{s.get("active",{})}\n更新时间：{s.get("updated_at","等待启动")}\n'
                      f'官方新增估算：${s.get("official_new_estimated_usd",0):.4f}（与前轮合计上限$5）')
        p = OUT / 'run.log'
        tail = p.read_text(encoding='utf-8', errors='replace').splitlines()[-15:] if p.exists() else []
        log.config(state='normal'); log.delete('1.0','end'); log.insert('end','\n'.join(tail)); log.config(state='disabled')
    except Exception as error:
        banner.config(text='读取状态暂不可用：'+type(error).__name__, bg='#fef3c7')
    root.after(3000, refresh)


refresh()
root.mainloop()
