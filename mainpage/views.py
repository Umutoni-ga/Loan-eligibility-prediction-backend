from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage

import joblib
import json
import pandas as pd

model=joblib.load('modelPipeline.pkl')

def predictionJson(request):
    data=json.loads(request.body)
    dataframe=pd.DataFrame({'x':data}).transpose()
    print(dataframe)
    prediction=model.predict(dataframe)
    prediction=str(prediction)
    
    return JsonResponse({'prediction': prediction})

def predictionFile(request):
    file=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(file.name, file)
    filePathName=fs.url(filePathName)
    filePath='.'+filePathName

    data=pd.read_csv(filePath)
    prediction=model.predict(data)
    
    prediction={i:j for i,j in zip(data['Loan_ID'], prediction)}

    return JsonResponse({'prediction': prediction})
    
