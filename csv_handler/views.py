import os
import shutil
from django.conf import settings
from django.shortcuts import redirect, render
from django.contrib import messages
import pandas as pd
from csv_handler.forms import UploadFileForm

def upload(request):

    form = UploadFileForm()

    if request.method == "POST":
        f = request.FILES["file"]

        if f.name.split(".")[-1] != "csv":
            messages.error(request, "The file should be of type csv!")
        
        else:

            if os.path.exists(settings.DATA_PATH):
                shutil.rmtree(settings.DATA_PATH)
            if os.path.exists(settings.MODELS_PATH):
                shutil.rmtree(settings.MODELS_PATH)
                
            os.mkdir(settings.DATA_PATH)
            os.mkdir(settings.MODELS_PATH)

            with open(settings.CSV_UPLOAD_PATH, "wb+") as destination:
                for chunk in f.chunks():
                    destination.write(chunk)
                        
            settings.DF = pd.read_csv(settings.CSV_UPLOAD_PATH, encoding="latin-1")

            messages.info(request, "File uploaded successfully")
            return redirect("/details")

    return render(request, "csv_handler/csv_upload.html", context={"form": form})