import socket
import threading
import time

HOST = "127.0.0.1"

PORT1 = 8001
PORT2 = 8002
PORT3 = 8003
PORT4 = 8004
PORT5 = 8005

conn1 = None
conn2 = None
conn3 = None
conn4 = None
conn5 = None

def tcp1():
    global conn1
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT1))
        s.listen(1)
        conn1, _ = s.accept()
    except:
        pass
def tcp2():
    global conn2
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT2))
        s.listen(1)
        conn2, _ = s.accept()
    except:
        pass
def tcp3():
    global conn3
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT3))
        s.listen(1)
        conn3, _ = s.accept()
    except:
        pass
def tcp4():
    global conn4
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT4))
        s.listen(1)
        conn4, _ = s.accept()
    except:
        pass
def tcp5():
    global conn5
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT5))
        s.listen(1)
        conn5, _ = s.accept()
    except:
        pass

threading.Thread(target=tcp1, daemon=True).start()
threading.Thread(target=tcp2, daemon=True).start()
threading.Thread(target=tcp3, daemon=True).start()
threading.Thread(target=tcp4, daemon=True).start()
threading.Thread(target=tcp5, daemon=True).start()

time.sleep(1)
print("✅ 5个TCP已启动，对应开关会在发现故障时候闭合！")

# ======================
# 【核心】直接发 0，不等！
# 开关1 一连接就闭合！
# ======================
while True:
    try:
        if conn1:
            conn1.send(bytes([0]))   # 开关1 永远 = 0（闭合）
        if conn2:
            conn2.send(bytes([1]))   # 永远打开
        if conn3:
            conn3.send(bytes([1]))
        if conn4:
            conn4.send(bytes([1]))
        if conn5:
            conn5.send(bytes([1]))
    except:
        pass
    time.sleep(0.01)
