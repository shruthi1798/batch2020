#!D:\Program files\python\python.exe

import cgi, cgitb 
import urllib.request
import json

# FieldStorage ka instance create karne ke liye
form = cgi.FieldStorage() 

# edhar se input value lega
score=form.getvalue('Store')
dept=form.getvalue('Dept')
date=form.getvalue('Date')
hol=form.getvalue('Isholiday')


print ("Content-type:text/html\r\n\r\n")

print ('<!DOCTYPE html><html lang="en"><head><title>Sales Forecasting</title><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css"><script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script><script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script></head><body><div class="container"><div class="jumbotron"><h2>Predicted Sales</ht><h1>')
data =  {

        "Inputs": {

                "input1":
                {
                    "ColumnNames": ["Store", "Dept", "Date", "IsHoliday"],
                    "Values": [ [ score, dept, date, hol ], [ score, dept, date, hol ], ]
                },        },
            "GlobalParameters": {
}
    }


body = str.encode(json.dumps(data))
url = 'https://ussouthcentral.services.azureml.net/workspaces/bd8d0af8e6114907947fe0b8e6eb9e9d/services/93083f92e4d54f3d9bcd9d4832f41945/execute?api-version=2.0&details=true'
api_key = '94wX2Fk+TtPdmzDT7PJBVSDERbIhaPK87m4IhObPtZjpWaayQ1Nj0xwU92wDqDpXYmaHzDen5f0BgC8iY5Qggg==' 
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers) 

try:
    response = urllib.request.urlopen(req)
    result = response.read()
    result1=result[435:446]
    print ("<h1>",result1,"</h1><strong>For</strong><br>")
 
   
except:
  print("An exception occurred")

print ('<div class="row"><div class="col-sm-2"><h5>Store : </h5></div><div class="col-sm-2">',score,'</div></div>') 
print ('<div class="row"><div class="col-sm-2"2<h5>Dept : </h5></div><div class="col-sm-2">',dept,'</div></div>')
print ('<div class="row"><div class="col-sm-2"><h5>Date : </h5></div><div class="col-sm-2">',date,'</div></div>')
print ('<div class="row"><div class="col-sm-2"><h5>IsHoliday : </h5></div><div class="col-sm-2">',hol,'</div></div>')
print ('</div></body>')
print ('</html>')
