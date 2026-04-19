from django import forms
from .models import Guide, Reservation


class GuideForm(forms.ModelForm):
    locations = forms.ChoiceField(
        choices=Guide.LOCATION_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    class Meta:
        model = Guide
        fields = ['name', 'cin', 'phone_number', 'picture', 'age', 'sex', 'living_place', 'locations']
        widgets = {
            'name':         forms.TextInput(attrs={'class': 'form-control'}),
            'cin':          forms.TextInput(attrs={'class': 'form-control'}),
            'phone_number': forms.TextInput(attrs={'class': 'form-control'}),
            'age':          forms.NumberInput(attrs={'class': 'form-control'}),
            'sex':          forms.Select(attrs={'class': 'form-control'}),
            'living_place': forms.TextInput(attrs={'class': 'form-control'}),
            'picture':      forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }


class ReservationForm(forms.ModelForm):
    tour_date = forms.DateField(
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )

    class Meta:
        model = Reservation
        fields = ['tourist_name', 'tourist_phone_number', 'tour_date', 'message']
        widgets = {
            'tourist_name':         forms.TextInput(attrs={'class': 'form-control'}),
            'tourist_phone_number': forms.TextInput(attrs={'class': 'form-control'}),
            'message':              forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }