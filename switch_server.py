# # switch_server.py
# import socket
# import threading
# import time
# from typing import List, Optional, Tuple
#
# class SwitchServer:
#     """多端口开关模拟服务器，可独立控制每个开关状态（0闭合，1断开）"""
#
#     def __init__(self, host: str = "127.0.0.1", ports: Tuple[int, ...] = (8001, 8002, 8003, 8004, 8005)):
#         """
#         初始化配置，但不启动服务。
#         :param host: 监听地址
#         :param ports: 端口元组，长度决定开关数量（默认5个）
#         """
#         self.host = host
#         self.ports = ports
#         self.num_switches = len(ports)
#
#         # 服务状态
#         self._servers: List[Optional[socket.socket]] = [None] * self.num_switches
#         self._conns: List[Optional[socket.socket]] = [None] * self.num_switches
#         self._accept_threads: List[threading.Thread] = []
#         self._sender_thread: Optional[threading.Thread] = None
#         self._stop_event: Optional[threading.Event] = None
#
#         # 开关状态（0闭合/1断开），默认全断开
#         self._states: List[int] = [1] * self.num_switches
#         self._state_lock = threading.Lock()
#
#     def _accept_connection(self, port: int, idx: int):
#         """在指定端口等待一个连接，并保存到 _conns[idx]"""
#         try:
#             s = socket.socket()
#             s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#             s.bind((self.host, port))
#             s.listen(1)
#             self._servers[idx] = s
#             conn, _ = s.accept()
#             self._conns[idx] = conn
#         except Exception:
#             pass
#
#     def _sender_loop(self):
#         """发送线程：循环读取当前状态并发送到对应连接"""
#         while not self._stop_event.is_set():
#             with self._state_lock:
#                 states_copy = self._states[:]
#             for idx, conn in enumerate(self._conns):
#                 if conn is not None:
#                     try:
#                         conn.send(bytes([states_copy[idx]]))
#                     except Exception:
#                         pass
#             time.sleep(0.01)
#
#     def start(self):
#         """启动所有服务（非阻塞）"""
#         self.stop()  # 确保之前已停止
#
#         self._stop_event = threading.Event()
#         self._servers = [None] * self.num_switches
#         self._conns = [None] * self.num_switches
#         self._accept_threads = []
#
#         for idx, port in enumerate(self.ports):
#             t = threading.Thread(target=self._accept_connection, args=(port, idx), daemon=True)
#             t.start()
#             self._accept_threads.append(t)
#
#         time.sleep(1)  # 等待 socket 就绪
#         print(f"✅ 已启动 {self.num_switches} 个开关服务，监听端口：{self.ports}")
#
#         self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
#         self._sender_thread.start()
#
#     def stop(self):
#         """停止所有服务，关闭连接和 socket"""
#         if self._stop_event:
#             self._stop_event.set()
#
#         for conn in self._conns:
#             if conn:
#                 try:
#                     conn.close()
#                 except Exception:
#                     pass
#         for srv in self._servers:
#             if srv:
#                 try:
#                     srv.close()
#                 except Exception:
#                     pass
#
#         if self._sender_thread and self._sender_thread.is_alive():
#             self._sender_thread.join(timeout=0.5)
#
#         self._servers = [None] * self.num_switches
#         self._conns = [None] * self.num_switches
#         self._accept_threads = []
#         self._sender_thread = None
#         self._stop_event = None
#
#         print("✅ 所有开关服务已停止")
#
#     def set_switches(self, s1: int, s2: int, s3: int, s4: int, s5: int):
#         """
#         设置 5 个开关的状态。
#         每个参数应为 0（闭合）或 1（断开）。
#         """
#         if self.num_switches != 5:
#             raise ValueError(f"当前开关数量为 {self.num_switches}，不是 5，请使用 set_states() 方法传入列表。")
#         self.set_states([s1, s2, s3, s4, s5])
#
#     def set_states(self, states: List[int]):
#         """通用状态设置，states 列表长度必须与开关数量一致"""
#         if len(states) != self.num_switches:
#             raise ValueError(f"状态长度 {len(states)} 与开关数量 {self.num_switches} 不匹配")
#         with self._state_lock:
#             self._states = [int(v) for v in states]
#
#     def get_states(self) -> List[int]:
#         """获取当前状态副本"""
#         with self._state_lock:
#             return self._states[:]

# switch_server.py
import socket
import threading
import time
from typing import List, Optional, Tuple

class SwitchServer:
    """多端口开关模拟服务器，可独立控制每个开关状态（0闭合，1断开）"""

    def __init__(self, host: str = "127.0.0.1", ports: Tuple[int, ...] = (8001, 8002, 8003, 8004, 8005)):
        self.host = host
        self.ports = ports
        self.num_switches = len(ports)

        self._servers: List[Optional[socket.socket]] = [None] * self.num_switches
        self._conns: List[Optional[socket.socket]] = [None] * self.num_switches
        self._accept_threads: List[threading.Thread] = []
        self._sender_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

        self._states: List[int] = [1] * self.num_switches
        self._state_lock = threading.Lock()

        # 标志：是否已经执行过连接后的切换序列（只执行一次）
        self._sequence_done = False
        self._sequence_lock = threading.Lock()

    def _accept_connection(self, port: int, idx: int):
        """在指定端口等待一个连接，并保存到 _conns[idx]；若是第一个端口，执行切换序列"""
        try:
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, port))
            s.listen(1)
            self._servers[idx] = s
            print(f"📡 端口 {port} 已就绪，等待 MATLAB 连接...")
            conn, addr = s.accept()
            self._conns[idx] = conn
            print(f"✅ 端口 {port} 收到连接来自 {addr}")

            # ==================== 方案A：连接后执行切换序列 ====================
            if idx == 0:   # 只对第一个端口（8001）执行
                with self._sequence_lock:
                    if not self._sequence_done:
                        self._sequence_done = True
                        print("🔄 执行开关切换序列（连接触发）...")
                        # 1. 全部断开
                        self.set_states([1, 1, 1, 1, 1])
                        print("   → 全部断开 (1,1,1,1,1)")
                        time.sleep(0.1)
                        # 2. 开关1闭合，其余断开
                        self.set_states([0, 1, 1, 1, 1])
                        print("   → 开关1闭合，其余断开 (0,1,1,1,1)")
        except Exception as e:
            print(f"❌ 端口 {port} 错误: {e}")

    def _sender_loop(self):
        while not self._stop_event.is_set():
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
        self.stop()
        self._stop_event = threading.Event()
        self._servers = [None] * self.num_switches
        self._conns = [None] * self.num_switches
        self._accept_threads = []

        for idx, port in enumerate(self.ports):
            t = threading.Thread(target=self._accept_connection, args=(port, idx), daemon=True)
            t.start()
            self._accept_threads.append(t)

        time.sleep(1)
        print(f"✅ 已启动 {self.num_switches} 个开关服务，监听端口：{self.ports}")

        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()

    def stop(self):
        if self._stop_event:
            self._stop_event.set()
        for conn in self._conns:
            if conn:
                try:
                    conn.close()
                except:
                    pass
        for srv in self._servers:
            if srv:
                try:
                    srv.close()
                except:
                    pass
        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=0.5)
        self._servers = [None] * self.num_switches
        self._conns = [None] * self.num_switches
        self._accept_threads = []
        self._sender_thread = None
        self._stop_event = None
        print("✅ 所有开关服务已停止")

    def set_switches(self, s1: int, s2: int, s3: int, s4: int, s5: int):
        if self.num_switches != 5:
            raise ValueError(f"开关数量为 {self.num_switches}，请使用 set_states()")
        self.set_states([s1, s2, s3, s4, s5])

    def set_states(self, states: List[int]):
        if len(states) != self.num_switches:
            raise ValueError(f"状态数量 {len(states)} != 开关数量 {self.num_switches}")
        with self._state_lock:
            self._states = [int(v) for v in states]

    def get_states(self) -> List[int]:
        with self._state_lock:
            return self._states[:]