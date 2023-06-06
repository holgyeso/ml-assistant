from django import forms
from django.forms import formset_factory

class FeatureForm(forms.Form):
    field_name = forms.CharField( widget=forms.TextInput(attrs={'readonly':'readonly'}))
    dtype = forms.CharField( widget=forms.TextInput(attrs={'readonly':'readonly'}))
    include = forms.BooleanField()
    normalize = forms.CharField(widget=forms.Select(choices=[
        ('categorical - pandas.get_dummies()', 'categorical - pandas.get_dummies()'),
        ('ordinal - sklearn.OrdinalEncoder', 'ordinal - sklearn.OrdinalEncoder'), 
        ('numerical - min_max', 'numerical - min_max'),
        ('numerical - standard_scaler', 'numerical - standard_scaler'),
        ('numerical - no encoding', 'numerical - no encoding')
        ]))
    
FeatureFormSet = formset_factory(FeatureForm)

class ModelForm(forms.Form):
    def __init__(self, *args, **kwargs):
        choices = kwargs.pop('cols_included')
        output_options = kwargs.pop('output_options')
        default_target = kwargs.pop("default_target_choice")
        super(ModelForm, self).__init__(*args, **kwargs)
        self.fields["target_column"] = forms.ChoiceField(choices=choices, initial=default_target, required=False)
        self.fields["prediction_class"] = forms.ChoiceField(choices=output_options, required=False)
        self.fields["cluster_nr"] = forms.IntegerField(min_value=2, initial=8) 

    model = forms.CharField(widget=forms.Select(choices=[
        ('classification', 'classification'), 
        ('regression', 'regression'),
        ('clustering', 'clustering')
        ]))
    
    algorithm_to_use = forms.ChoiceField(choices=( 
        ("logistic regression", "logistic regression"),
        ("decision tree", "decision tree")
    )) # by default showing sub-models for regression


class PredictionForm(forms.Form):
    def __init__(self, *args, **kwargs):
        field_definition = kwargs.pop("field_definition")
        super(PredictionForm, self).__init__(*args, **kwargs)

        for col in field_definition:
            self.fields[col] = field_definition[col]
