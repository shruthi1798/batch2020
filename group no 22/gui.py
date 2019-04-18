import tkinter as tk
from tkinter import messagebox
master = tk.Tk()

def func():
#    print(var1, var2, var3)
    v1 = var1.get()
    v2 = var2.get()
    v3 = var3.get()
    print(v1, v2, v3)
    #tk.messagebox.showinfo("Severity", "Severity = 3")
    if((v1==1 or v1==2) and (v2==10 or v2==20)and (v3==100 or v3==200)):
       tk.messagebox.showinfo("Severity", "Severity = 3")
    elif((v1==3 or v1==4 or v1==7) and (v2==30 or v2==40)and (v3==300)):
       tk.messagebox.showinfo("Severity", "Severity = 2")
    elif((v1==5 or v1==6 or v1==8) and (v2==50)and (v3==400 or v3==500)):
       tk.messagebox.showinfo("Severity", "Severity = 3")
    else:
        tk.messagebox.showinfo("Severity", "Severity = 2")
    
#fr= tk.parent_window.maxsize(x,x)
var1 = tk.IntVar()
tk.Label(master, text='Weather conditions').grid(row=0)
tk.Radiobutton(master, text='Fine without high winds', variable=var1, value=1).grid(row=0, column=1, sticky='W') 
tk.Radiobutton(master, text='Raining without high winds', variable=var1, value=2).grid(row=1, column=1, sticky='W')
tk.Radiobutton(master, text='Fine with high winds', variable=var1, value=3).grid(row=2, column=1, sticky='W') 
tk.Radiobutton(master, text='Raining with high winds', variable=var1, value=4).grid(row=3, column=1, sticky='W')
tk.Radiobutton(master, text='Snowing without high winds', variable=var1, value=5).grid(row=4, column=1, sticky='W') 
tk.Radiobutton(master, text='Snowing with high winds', variable=var1, value=6).grid(row=5, column=1, sticky='W')
tk.Radiobutton(master, text='other', variable=var1, value=7).grid(row=6, column=1, sticky='W')
tk.Radiobutton(master, text='Fog or mist', variable=var1, value=8).grid(row=6, column=2, sticky='W') 
var1.get()
var2 = tk.IntVar() 
tk.Label(master, text='Road Type').grid(row=7)
tk.Radiobutton(master, text='Single carriageway', variable=var2, value=10).grid(row=8, column=1, sticky='W') 
tk.Radiobutton(master, text='Dual carriageway', variable=var2, value=20).grid(row=9, column=1, sticky='W')
tk.Radiobutton(master, text='Roundabout', variable=var2, value=30).grid(row=10, column=1, sticky='W') 
tk.Radiobutton(master, text='One way street', variable=var2, value=40).grid(row=11, column=1, sticky='W')
tk.Radiobutton(master, text='Slip road', variable=var2, value=50).grid(row=12, column=1, sticky='W')
var2.get()
var3 = tk.IntVar() 
tk.Label(master, text='Light Conditions').grid(row=13)
tk.Radiobutton(master, text='Daylight: Street light present', variable=var3, value=100).grid(row=14, column=1, sticky='W') 
tk.Radiobutton(master, text='Darkness: Street lights present and lit', variable=var3, value=200).grid(row=15, column=1, sticky='W')
tk.Radiobutton(master, text='Darkness: No Street lights', variable=var3, value=300).grid(row=16, column=1, sticky='W')
tk.Radiobutton(master, text='Darkness: Street lights unknown', variable=var3, value=400).grid(row=17, column=1, sticky='W')
tk.Radiobutton(master, text='Darkness: Street lights present but unlit', variable=var3, value=500).grid(row=18, column=1, sticky='W')
var3.get()
'''tk.Label(master, text='Time').grid(row=4) 
e1 = tk.Entry(master) 
e2 = tk.Entry(master) 
e1.grid(row=3, column=1) 
e2.grid(row=4, column=1)'''
B = tk.Button(text="submit", command=func).grid(row=20)
#, command=lambda fname.py
tk.mainloop() 
