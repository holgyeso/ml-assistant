from django import forms
from django.conf import settings

class InspectDataForm(forms.Form):
    order = forms.CharField(widget=forms.Select(choices=[('first', 'first'), ('last', 'last')]))

    nr = forms.DecimalField(min_value=0,
                            step_size=1,
                            widget=forms.NumberInput(attrs={'placeholder': 'n'}))