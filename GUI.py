
import tkinter as tk
import tkinter.filedialog
import functions
import threading
from queue import Queue

def selectpath1():
    global Path1
    path_=tkinter.filedialog.askopenfilename()
    Path1=path_
    path1.set(path_)

def selectpath2():
    global Path2

    path_=tkinter.filedialog.askopenfilename()
    Path2 = path_
    path2.set(path_)
def selectpath3():
    global Path3

    path_=tkinter.filedialog.askopenfilename()
    Path3 = path_
    path3.set(path_)
def selectpath4():
    global Path4

    path_=tkinter.filedialog.askopenfilename()
    Path4 = path_
    path4.set(path_)
def selectpath5():
    global Path5
    path_=tkinter.filedialog.askopenfilename()
    Path5 = path_
    path5.set(path_)
def run(Path1,Path2):
    Q1=Queue()
    Q2=Queue()
    t=threading.Thread(target=functions.pref,args=(Path1,Path2,Q1,Q2))
    t.start()
    t.join()
    lo1='xxx111.xlsx';lo2='xxx222.xlsx'
    Q31=Queue()
    Q32=Queue()
    Q41=Queue()
    Q42=Queue()
    t1=threading.Thread(target=functions.pre1,args=(Q1.get(),lo1,Q31,Q32))
    t2= threading.Thread(target=functions.pre1,args=(Q2.get(),lo2,Q41,Q42))
    t2.start()
    t2.join()
    t1.start()
    t1.join()
    Q5=Queue()
    t4=threading.Thread(target=functions.id,args=(Q31.get(),Q41.get(),Q5))
    t4.start()
    t4.join()
    q1=Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()
    t5=threading.Thread(target=functions.www,args=(Q5.get(),Q32.get(),q1,q2,q3,q4))
    t5.start()
    t5.join()
    t6=threading.Thread(target=functions.predict,args=(q1.get(),q2.get(),q3.get(),q4.get(),Path4))
    t6.start()
    t6.join()
def run2(Path3):
    lo1='zzzz.xlsx'
    t1=threading.Thread(target=functions.pre3,args=(Path3,lo1))
    t1.start()
    t1.join()
def run3(Path5):
    Firm=firm.get()
    print(Firm)
    t1=threading.Thread(target=functions.radar,args=(Path5,Firm))
    t1.start()
    t1.join()
box=tk.Tk()
box.title('公司评分预测程序')
path1=tk.StringVar()
path2=tk.StringVar()
path3=tk.StringVar()
path4=tk.StringVar()
path5=tk.StringVar()
firm=tk.StringVar()
Path1='asd'
Path2='asd'
Path3='asd'
Path4='asd'
Path5='asd'
tk.Label(box,text='功能3：预测公司的分数走势').grid(row=0,column=7)
tk.Label(box,text='模拟文件1：').grid(row=1,column=6)
tk.Label(box,text='模拟文件2：').grid(row=2,column=6)
tk.Label(box,text='预测文件：').grid(row=3,column=6)
e1=tk.Entry(box,textvariable=path1).grid(row=1,column=7)
e2=tk.Entry(box,textvariable=path2).grid(row=2,column=7)
tk.Entry(box,textvariable=path4).grid(row=3,column=7)
b1=tk.Button(box,text='select',command=selectpath1).grid(row=1,column=8)
b2=tk.Button(box,text='select',command=selectpath2).grid(row=2,column=8)
tk.Button(box,text='select',command=selectpath4).grid(row=3,column=8)
tk.Button(box,text='run',command=lambda :run(Path1,Path2)).grid(row=4,column=8)

tk.Label(box,text='功能1：计算公司的分数').grid(row=0,column=1)
tk.Label(box,text='计算分数文件：').grid(row=1,column=0)
tk.Entry(box,textvariable=path3).grid(row=1,column=1)
tk.Button(box,text='select',command=selectpath3).grid(row=1,column=2)
tk.Button(box,text='run',command=lambda : run2(Path3)).grid(row=2,column=2)

tk.Label(box,text='功能2：公司一级指数雷达图').grid(row=0,column=4)
tk.Label(box,text='数据文件：').grid(row=1,column=3)
tk.Entry(box,textvariable=path5).grid(row=1,column=4)
tk.Button(box,text='select',command=selectpath5).grid(row=1,column=5)
tk.Label(box,text='公司名称：').grid(row=2,column=3)
tk.Entry(box,textvariable=firm).grid(row=2,column=4)

tk.Button(box,text='run',command=lambda : run3(Path5)).grid(row=2,column=5)
box.mainloop()