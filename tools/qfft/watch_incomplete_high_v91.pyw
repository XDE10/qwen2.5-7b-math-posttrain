"""Read-only status window for the 62-question high supplement."""
from pathlib import Path
import json
import tkinter as tk
from tkinter import ttk
OUT = Path(__file__).resolve().parents[1] / 'artifacts/qfft_incomplete_high_v91'
root = tk.Tk(); root.title('项目一 | 62题 high 抗重复补跑 v91'); root.geometry('850x530')
tk.Label(root,text='62题补跑 · high＋抗重复提示词 · 2题并发',font=('Microsoft YaHei',17,'bold')).pack(pady=15)
tk.Label(root,text='每题一次；原1000份和已有27份修复候选不覆盖。关闭窗口不会停止任务。').pack()
banner=tk.Label(root,pady=12,font=('Microsoft YaHei',12));banner.pack(fill='x',padx=15,pady=10)
bar=ttk.Progressbar(root,maximum=62,length=780);bar.pack()
detail=tk.Label(root,justify='left',anchor='w',font=('Microsoft YaHei',11));detail.pack(fill='x',padx=20,pady=10)
log=tk.Text(root,height=12,bg='#111827',fg='white');log.pack(fill='both',expand=True,padx=15,pady=10)
def refresh():
    try:
        p=OUT/'status.json';s=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        state=s.get('state','starting');banner.config(text=f'状态：{state} | 完成 {s.get("completed",0)}/62',bg='#fee2e2' if state=='paused_requires_review' else '#dcfce7')
        bar['value']=s.get('completed',0)
        detail.config(text=f'完整≤16K：{s.get("complete_within_16k",0)}    完整超16K：{s.get("complete_over_16k",0)}    仍不完整：{s.get("incomplete",0)}    连接未知：{s.get("transport_unknown_skip",0)}\n当前题目：{s.get("active",[])}\n更新时间：{s.get("updated_at","等待启动")}\n官方本轮估算：${s.get("official_new_estimated_usd",0):.4f}（跨轮累计上限$5）')
        p=OUT/'run.log';tail=p.read_text(encoding='utf-8',errors='replace').splitlines()[-15:] if p.exists() else []
        log.config(state='normal');log.delete('1.0','end');log.insert('end','\n'.join(tail));log.config(state='disabled')
    except Exception as error:banner.config(text='读取状态暂不可用：'+type(error).__name__,bg='#fef3c7')
    root.after(3000,refresh)
refresh();root.mainloop()
