from tkinter import *
from naive import naive
from naive import accur
import time

root = Tk()
root.title("Review analyser")
root.configure(background='#C4C6C6')

status_label =Label(root, bg="#C4C6C6",text ="")
status_label.grid(row =4, column =1, pady=10, padx=5)

label1 =Label(root, text ="Review", bg="#C4C6C6", font="Times 24 italic bold")
label1.grid(row =1, column =0, pady=10, padx=5)

#entryFrame = Frame(root, width=454, height=20)
#entryFrame.grid(row=1, column=1)

num1_txtbx =Entry(root,textvariable=1,width=60, bd=3, font="Times 20 italic", relief=RAISED)
num1_txtbx.grid(row =1, column =1, padx=5)
num1 =num1_txtbx.get()
def compute():
	if (num1_txtbx.get()!= ""):
		num1 =num1_txtbx.get()
		answer=naive(num1)
		if answer==1:
			answer_label.configure(text ="Thank you for a positive review", font="Times 24 italic")
		else:
			answer_label.configure(text ="It was a negative review.", font="Times 24 italic")
		status_label.configure(text ="Successfully Computed", font="Times 24 italic")



calculate_button =Button(root, text="Submit here", bg="#51C21F", command= compute, font="Times 24 italic")
calculate_button.grid(row =3, column =0, columnspan =2, pady=10, padx=5)
"""
cal_button =Button(root, text="Aboutaccuracy")
cal_button.grid(row =4, column =0, columnspan =2)

"""
answer_label =Label(root, height =10, width =50, bg ="black", fg ="#00FF00", text ="Enter review for analysis", wraplength =500, font="Times 24 italic")
answer_label.grid(row =5, column =0, columnspan =2, pady=10, padx=5)

root.mainloop()