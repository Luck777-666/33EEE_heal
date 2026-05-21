# switch_server.py
import socket
import threading
import time
from typing import List, Optional, Tuple

class SwitchServer:
    """多端口开关模拟服务器，可独立控制每个开关状态（0闭合，1断开）"""

    def __init__(self, host: str = "127.0.0.1", ports: Tuple[int, ...] = (8001, 8002, 8003, 8004, 8005)):
        """
        初始化配置，但不启动服务。
        :param host: 监听地址
        :param ports: 端口元组，长度决定开关数量（默认5个）
        """
        self.host = host
        self.ports = ports
        self.num_switches = len(ports)

        # 服务状态
        self._servers: List[Optional[socket.socket]] = [None] * self.num_switches
        self._conns: List[Optional[socket.socket]] = [None] * self.num_switches
        self._accept_threads: List[threading.Thread] = []
        self._sender_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

        # 开关状态（0闭合/1断开），默认全断开
        self._states: List[int] = [1] * self.num_switches
        self._state_lock = threading.Lock()

    def _accept_connection(self, port: int, idx: int):
        """在指定端口等待一个连接，并保存到 _conns[idx]"""
        try:
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, port))
            s.listen(1)
            self._servers[idx] = s
            conn, _ = s.accept()
            self._conns[idx] = conn
        except Exception:
            pass

    def _sender_loop(self):
        """发送线程：循环读取当前状态并发送到对应连接"""
        while not self._stop_event.is_set():
            # 复制当前状态快照，减少锁持有时间
            with self._state_lock:
                states_copy = self._states[:]
            for idx, conn in enumerate(self._conns):
                if conn is not None:
                    try:
                        conn.send(bytes([states_copy[idx]]))
                    except Exception:
                        pass
            time.sleep(0.01)

    def start(self):
        """启动所有服务（非阻塞）"""
        self.stop()  # 确保之前已停止

        self._stop_event = threading.Event()
        self._servers = [None] * self.num_switches
        self._conns = [None] * self.num_switches
        self._accept_threads = []

        # 启动每个端口的 accept 线程
        for idx, port in enumerate(self.ports):
            t = threading.Thread(target=self._accept_connection, args=(port, idx), daemon=True)
            t.start()
            self._accept_threads.append(t)

        time.sleep(1)  # 等待 socket 就绪
        print(f"✅ 已启动 {self.num_switches} 个开关服务，监听端口：{self.ports}")

        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()

    def stop(self):
        """停止所有服务，关闭连接和 socket"""
        if self._stop_event:
            self._stop_event.set()

        # 关闭所有客户端连接
        for conn in self._conns:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
        # 关闭所有服务端 socket
        for srv in self._servers:
            if srv:
                try:
                    srv.close()
                except Exception:
                    pass

        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=0.5)

        # 重置内部引用
        self._servers = [None] * self.num_switches
        self._conns = [None] * self.num_switches
        self._accept_threads = []
        self._sender_thread = None
        self._stop_event = None

        print("✅ 所有开关服务已停止")

    def set_switches(self, s1: int, s2: int, s3: int, s4: int, s5: int):
        """
        设置 5 个开关的状态。
        每个参数应为 0（闭合）或 1（断开）。
        如果开关数量不是5，请直接使用 set_states(list)。
        """
        if self.num_switches != 5:
            raise ValueError(f"当前开关数量为 {self.num_switches}，不是 5，请使用 set_states() 方法传入列表。")
        self.set_states([s1, s2, s3, s4, s5])

    def set_states(self, states: List[int]):
        """通用状态设置，states 列表长度必须与开关数量一致"""
        if len(states) != self.num_switches:
            raise ValueError(f"状态长度 {len(states)} 与开关数量 {self.num_switches} 不匹配")
        with self._state_lock:
            self._states = [int(v) for v in states]  # 确保为整数

    def get_states(self) -> List[int]:
        """获取当前状态副本"""
        with self._state_lock:
            return self._states[:]