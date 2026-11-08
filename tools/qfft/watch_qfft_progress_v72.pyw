"""Read-only native progress window; closing this window does not stop generation."""
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import psutil

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts/qfft_ds_dataset_v66'
CONTROL = ROOT / 'artifacts/qfft_all_supervisor_v71'
TITLE = '项目一 | 1000题造数实时监控 v73'


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {} if default is None else default


def process_for(filename):
    result = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' not in (proc.info['name'] or '').lower():
                continue
            if any(Path(arg).name.lower() == filename for arg in (proc.info['cmdline'] or [])):
                result.append(proc.info['pid'])
        except (psutil.Error, OSError, ValueError):
            pass
    return result


def tail(path, size=10000):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return '等待日志文件……'


class Monitor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(TITLE)
        self.root.geometry('1000x760')
        self.root.minsize(880, 650)
        self.root.configure(bg='#f3f5f8')
        self.cache = {}
        self.last_text = None
        self.last_alarm = None
        self.error_since = None
        style = ttk.Style(self.root)
        if 'vista' in style.theme_names():
            style.theme_use('vista')
        style.configure('TLabel', font=('Microsoft YaHei UI', 10))
        style.configure('TButton', font=('Microsoft YaHei UI', 10))
        body = ttk.Frame(self.root, padding=20)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text='项目一 · 固定1000题生成', font=('Microsoft YaHei UI', 18, 'bold')).pack(anchor='w')
        ttk.Label(body, text='只读监控，每5秒刷新；关闭本窗口不会停止后台任务。').pack(anchor='w', pady=(5, 14))
        self.banner = tk.Label(body, text='正在读取实际进度……', anchor='w', padx=14, pady=12,
                               font=('Microsoft YaHei UI', 12, 'bold'), bg='#e8eefb', fg='#163c78')
        self.banner.pack(fill='x')
        self.progress_text = tk.StringVar()
        ttk.Label(body, textvariable=self.progress_text, font=('Microsoft YaHei UI', 16, 'bold')).pack(anchor='w', pady=(14, 7))
        self.progress = ttk.Progressbar(body, maximum=1000, mode='determinate')
        self.progress.pack(fill='x')
        self.details = tk.StringVar()
        ttk.Label(body, textvariable=self.details, justify='left', wraplength=930).pack(anchor='w', pady=(12, 6))
        self.inflight = tk.StringVar()
        ttk.Label(body, textvariable=self.inflight, justify='left', wraplength=930).pack(anchor='w', pady=(0, 10))
        self.clock_text = tk.StringVar()
        ttk.Label(body, textvariable=self.clock_text).pack(anchor='w', pady=(0, 8))
        controls = ttk.Frame(body)
        controls.pack(fill='x', pady=(0, 10))
        ttk.Button(controls, text='立即刷新', command=self.refresh).pack(side='left')
        ttk.Button(controls, text='打开日志目录', command=lambda: os.startfile(CONTROL)).pack(side='left', padx=10)
        self.pin = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text='窗口置顶', variable=self.pin,
                        command=lambda: self.root.attributes('-topmost', self.pin.get())).pack(side='left', padx=10)
        self.beep = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text='异常时提示音', variable=self.beep).pack(side='left')
        ttk.Label(body, text='最近运行日志（出错后保留，不会自动关闭）', font=('Microsoft YaHei UI', 10, 'bold')).pack(anchor='w')
        self.log = ScrolledText(body, height=13, wrap='word', font=('Consolas', 10), bg='#111827', fg='#e5e7eb',
                                insertbackground='white', padx=10, pady=10)
        self.log.pack(fill='both', expand=True, pady=(5, 0))
        self.log.configure(state='disabled')
        self.refresh()
        self.root.after(5000, self.tick)

    def refresh(self):
        try:
            self.render()
        except Exception as error:
            self.banner.configure(text='监控读取异常：' + type(error).__name__ + '；后台任务未被本窗口操作。',
                                  bg='#fee2e2', fg='#991b1b')
            (CONTROL / 'monitor_error.txt').write_text(repr(error), encoding='utf-8')

    def render(self):
        current_paths = sorted((DATA / 'candidates').glob('*.json'))
        latest = 0.0
        for path in current_paths:
            stat = path.stat()
            latest = max(latest, stat.st_mtime)
            cached = self.cache.get(path.name)
            if cached and cached['mtime'] == stat.st_mtime_ns:
                continue
            value = read_json(path)
            if 'index' in value:
                self.cache[path.name] = {'mtime': stat.st_mtime_ns, 'index': value['index'],
                                        'length_class': value.get('length_class', 'unknown'),
                                        'provider': value.get('provider', 'unknown')}
        # Only count files that currently exist, not removed cached entries.
        rows = [self.cache[p.name] for p in current_paths if p.name in self.cache]
        counts = Counter(r['length_class'] for r in rows)
        providers = Counter(r['provider'] for r in rows)
        supervisor = process_for('continue_qfft_all_v71.py') + process_for('start_qfft_supervisor_v73.pyw')
        worker = process_for('build_qfft_ds_dataset_v66.py')
        state = read_json(CONTROL / 'status.json')
        routing = read_json(DATA / 'routing.json')
        stage = state.get('state', 'unknown')
        if stage == 'generation_complete' and len(rows) == 1000:
            message, colour, foreground, alarm = '已完成1000条生成；数学质量仍待审查，未自动入训', '#dcfce7', '#166534', False
        elif stage == 'finished_with_deferred_requests':
            message, colour, foreground, alarm = f'可继续的题目已处理完：{len(rows)}条保存；{1000-len(rows)}条未知请求待单独处理', '#fef3c7', '#92400e', False
        elif not supervisor:
            message, colour, foreground, alarm = '后台监督进程已停止！请查看下方错误日志，并让助手检查后恢复。', '#fee2e2', '#991b1b', True
        elif stage in ('stopped_requires_review', 'supervisor_error'):
            message, colour, foreground, alarm = '任务异常暂停，需要检查；不会自动重复付费请求。', '#fee2e2', '#991b1b', True
        elif stage in ('waiting_offpeak', 'paused_for_peak'):
            message, colour, foreground, alarm = '按低价门禁暂停；计划恢复：' + state.get('resume_at', '下一低价窗口'), '#fef3c7', '#92400e', False
        elif worker:
            message, colour, foreground, alarm = '正在后台生成 · 任务进程存活', '#dcfce7', '#166534', False
        else:
            message, colour, foreground, alarm = '监督进程存活 · 正在归档或切换阶段，请查看日志', '#e8eefb', '#163c78', False
        self.banner.configure(text=message, bg=colour, fg=foreground)
        if alarm and self.last_alarm != (stage, tuple(supervisor)):
            if self.beep.get():
                self.root.bell()
            self.last_alarm = (stage, tuple(supervisor))
        if not alarm:
            self.last_alarm = None
        self.progress_text.set(f'已保存 {len(rows)} / 1000 题    {len(rows) / 10:.1f}%    剩余 {max(0, 1000-len(rows))} 题')
        self.progress['value'] = len(rows)
        complete_long = counts['complete_16k_to_32k'] + counts['complete_over_32k_student_tokens']
        self.details.set(
            f'完整且≤16K：{counts["complete_within_16k"]}   完整超16K：{complete_long}   结构未完成：{counts["incomplete"]}\n'
            f'Go：{providers["go"]}   官方：{providers["official"]}   官方累计估算：${routing.get("official_estimated_usd", 0):.4f} / $5.00\n'
            f'监督PID：{", ".join(map(str, supervisor)) or "不在运行"}   生成PID：{", ".join(map(str, worker)) or "不在运行"}\n'
            f'中断后未知请求暂缓：{len(read_json(DATA / "reviewed_unresolved_v73.json").get("records", []))}题（不自动重发）')
        pending = []
        for request in DATA.glob('calls/*/*/request.json'):
            if request.with_name('response.json').exists() or request.with_name('error.json').exists():
                continue
            deferred = read_json(DATA / 'reviewed_unresolved_v73.json').get('records', [])
            if request.parent.name == 'go' and int(request.parent.parent.name) in {r['index'] for r in deferred}:
                continue
            dispatch = read_json(request)
            start = dispatch.get('started_utc')
            try:
                elapsed = max(0, time.time() - datetime.fromisoformat(start).timestamp())
                duration = f'{int(elapsed)//60}分{int(elapsed)%60:02d}秒'
            except (ValueError, TypeError):
                duration = '未知'
            pending.append(f'{request.parent.parent.name} ({request.parent.name}, {dispatch.get("body", {}).get("reasoning_effort", "?")}, {duration})')
        self.inflight.set('在途请求：' + ('；'.join(pending) if pending else '当前没有未返回的请求'))
        self.clock_text.set('监控刷新：' + datetime.now().strftime('%H:%M:%S') + '    最近保存：' +
                            (datetime.fromtimestamp(latest).strftime('%H:%M:%S') if latest else '尚无') +
                            '    工作日14–18点不派新请求')
        relative_log = state.get('log_path')
        log_path = ROOT / relative_log if relative_log else CONTROL / 'supervisor_stderr.log'
        text = tail(log_path)
        if alarm:
            errors = CONTROL / 'supervisor_v73_stderr.log'
            text += '\n\n[监督器错误日志]\n' + tail(errors if errors.exists() else CONTROL / 'supervisor_stderr.log', 5000)
            pause = read_json(DATA / 'last_pause.json')
            if pause:
                text += '\n\n[最后暂停记录；请核对时间]\n' + json.dumps(pause, ensure_ascii=False, indent=2)
        if text != self.last_text:
            self.log.configure(state='normal')
            self.log.delete('1.0', 'end')
            self.log.insert('end', text)
            self.log.configure(state='disabled')
            self.log.see('end')
            self.last_text = text
        # UI receipt enables checking rendering/refresh without controlling the app.
        save = {'window_title': TITLE, 'refreshed_at': datetime.now().isoformat(),
                'saved_candidates': len(rows), 'supervisor_pids': supervisor, 'worker_pids': worker,
                'display_status': message, 'alarm': alarm}
        (CONTROL / 'monitor_status.json').write_text(json.dumps(save, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    def tick(self):
        self.refresh()
        self.root.after(5000, self.tick)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    Monitor().run()
