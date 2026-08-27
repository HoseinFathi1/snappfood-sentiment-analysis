from django.shortcuts import render

from . import model_loader


def classify_view(request):
    """Renders the form; on POST, runs the model and shows the result."""
    result = None
    submitted_text = ''

    if request.method == 'POST':
        submitted_text = request.POST.get('text', '').strip()
        if submitted_text:
            result = model_loader.predict_sentiment(submitted_text)

    return render(request, 'classifier/index.html', {
        'result': result,
        'submitted_text': submitted_text,
    })
