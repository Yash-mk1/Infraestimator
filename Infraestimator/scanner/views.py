"""
scanner/views.py — MODIFIED
Uses Cloudinary for image storage instead of local disk.
"""

import os
import base64
import uuid

import cv2
import numpy as np
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .analyser import InfrastructureAnalyzer, HealthScorer
from .models import ScanResult
from .image_validator import validate_image
from .cloudinary_helper import upload_and_save_local


# ── Public pages ──────────────────────────────────────────────────────────────
def home(request):
    return render(request, 'scanner/home.html')

def analyse(request):
    return render(request, 'scanner/analyse.html')

def how_it_works(request):
    return render(request, 'scanner/how_it_works.html')

def technology(request):
    return render(request, 'scanner/technology.html')

def docs(request):
    return render(request, 'scanner/docs.html')


# ── Auth ──────────────────────────────────────────────────────────────────────
def signup_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        username  = request.POST.get('username', '').strip()
        email     = request.POST.get('email', '').strip()
        password1 = request.POST.get('password1', '')
        password2 = request.POST.get('password2', '')

        if not username or not password1:
            messages.error(request, 'Username and password are required.')
        elif password1 != password2:
            messages.error(request, 'Passwords do not match.')
        elif len(password1) < 6:
            messages.error(request, 'Password must be at least 6 characters.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken.')
        else:
            user = User.objects.create_user(
                username=username, email=email, password=password1
            )
            login(request, user)
            messages.success(request, f'Welcome, {username}!')
            return redirect('home')

    return render(request, 'scanner/signup.html')


def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user     = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            next_url = request.POST.get('next') or request.GET.get('next') or 'home'
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid username or password.')

    return render(request, 'scanner/login.html', {
        'next': request.GET.get('next', '')
    })


def logout_view(request):
    logout(request)
    return redirect('home')


# ── Scan ──────────────────────────────────────────────────────────────────────
@login_required(login_url='/login/')
def scan(request):
    if request.method != 'POST':
        return redirect('analyse')

    material  = request.POST.get('material', 'general')
    image_bgr = None

    # ── Decode image ──────────────────────────────────────────────────────────
    if 'image_file' in request.FILES:
        file_bytes = np.frombuffer(
            request.FILES['image_file'].read(), dtype=np.uint8
        )
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif request.POST.get('image_b64'):
        b64 = request.POST['image_b64']
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        image_bgr = cv2.imdecode(
            np.frombuffer(base64.b64decode(b64), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

    if image_bgr is None:
        return render(request, 'scanner/analyse.html',
                      {'error': 'Could not decode image. Please try again.'})

    # ── Validate image ────────────────────────────────────────────────────────
    is_valid, reason, confidence = validate_image(image_bgr)
    if not is_valid:
        return render(request, 'scanner/analyse.html', {
            'error':                 reason,
            'validation_failed':     True,
            'validation_confidence': round(confidence * 100, 1),
        })

    # ── Resize if needed ──────────────────────────────────────────────────────
    h, w = image_bgr.shape[:2]
    if max(h, w) > 1600:
        scale     = 1600 / max(h, w)
        image_bgr = cv2.resize(
            image_bgr, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    # ── Run analysis ──────────────────────────────────────────────────────────
    analyzer  = InfrastructureAnalyzer()
    scorer    = HealthScorer()
    detection = analyzer.analyze(image_bgr)
    report    = scorer.score(detection, material)

    # ── Upload images to Cloudinary (or local fallback) ───────────────────────
    uid          = uuid.uuid4().hex[:10]
    results_dir  = os.path.join(settings.MEDIA_ROOT, 'results')
    media_prefix = settings.MEDIA_URL + 'results/'

    orig_name      = f'original_{uid}.jpg'
    annotated_name = f'annotated_{uid}.jpg'
    heatmap_name   = f'heatmap_{uid}.jpg'

    orig_url = upload_and_save_local(
        image_bgr,
        orig_name,
        os.path.join(results_dir, orig_name),
        media_prefix
    )
    annotated_url = upload_and_save_local(
        detection.annotated_image,
        annotated_name,
        os.path.join(results_dir, annotated_name),
        media_prefix
    )
    heatmap_url = upload_and_save_local(
        detection.heatmap_image,
        heatmap_name,
        os.path.join(results_dir, heatmap_name),
        media_prefix
    )

    # ── Save to database ──────────────────────────────────────────────────────
    scan_obj = ScanResult.objects.create(
        user                 = request.user,
        material             = material,
        health_score         = report.health_score,
        condition            = report.condition,
        estimated_life_years = report.estimated_life_years,
        critical             = report.critical,
        crack_score          = round(detection.crack_score, 2),
        seep_score           = round(detection.seep_score, 2),
        surface_score        = round(detection.surface_score, 2),
        crack_area_pct       = round(detection.crack_area_pct, 3),
        seep_area_pct        = round(detection.seep_area_pct, 3),
        num_crack_regions    = detection.num_crack_regions,
        num_seep_regions     = detection.num_seep_regions,
        largest_crack_mm_eq  = detection.largest_crack_mm_eq,
        ai_used              = detection.ai_used,
        ai_confidence        = detection.ai_confidence,
        warnings             = '\n'.join(report.warnings),
        recommendations      = '\n'.join(report.recommendations),
        original_url         = orig_url,
        annotated_url        = annotated_url,
        heatmap_url          = heatmap_url,
    )

    # ── Store in session ──────────────────────────────────────────────────────
    request.session['report'] = {
        'scan_id':               scan_obj.pk,
        'timestamp':             report.timestamp,
        'health_score':          report.health_score,
        'condition':             report.condition,
        'estimated_life_years':  report.estimated_life_years,
        'critical':              report.critical,
        'warnings':              report.warnings,
        'recommendations':       report.recommendations,
        'material':              material,
        'crack_score':           round(detection.crack_score, 2),
        'seep_score':            round(detection.seep_score, 2),
        'surface_score':         round(detection.surface_score, 2),
        'crack_area_pct':        round(detection.crack_area_pct, 3),
        'seep_area_pct':         round(detection.seep_area_pct, 3),
        'num_crack_regions':     detection.num_crack_regions,
        'num_seep_regions':      detection.num_seep_regions,
        'largest_crack_mm_eq':   detection.largest_crack_mm_eq,
        'ai_used':               detection.ai_used,
        'ai_confidence':         round(detection.ai_confidence * 100, 1),
        'annotated_url':         annotated_url,
        'heatmap_url':           heatmap_url,
        'original_url':          orig_url,
        'validation_confidence': round(confidence * 100, 1),
    }

    return redirect('result')


# ── Result ────────────────────────────────────────────────────────────────────
@login_required(login_url='/login/')
def result(request):
    report = request.session.get('report')
    if not report:
        return redirect('analyse')

    score = report['health_score']
    if   score >= 8.5: score_class = 'excellent'
    elif score >= 7.0: score_class = 'good'
    elif score >= 5.5: score_class = 'fair'
    elif score >= 4.0: score_class = 'poor'
    elif score >= 2.5: score_class = 'critical'
    else:              score_class = 'failure'

    return render(request, 'scanner/result.html', {
        'report':      report,
        'score_class': score_class,
        'score_pct':   int((report['health_score'] / 10) * 100),
    })


# ── History ───────────────────────────────────────────────────────────────────
@login_required(login_url='/login/')
def history(request):
    scans = ScanResult.objects.filter(
        user=request.user
    ).order_by('-created_at')
    return render(request, 'scanner/history.html', {'scans': scans})


@login_required(login_url='/login/')
def history_detail(request, scan_id):
    try:
        scan = ScanResult.objects.get(pk=scan_id, user=request.user)
    except ScanResult.DoesNotExist:
        return redirect('history')

    score = scan.health_score
    if   score >= 8.5: score_class = 'excellent'
    elif score >= 7.0: score_class = 'good'
    elif score >= 5.5: score_class = 'fair'
    elif score >= 4.0: score_class = 'poor'
    elif score >= 2.5: score_class = 'critical'
    else:              score_class = 'failure'

    report = {
        'scan_id':               scan.pk,
        'timestamp':             scan.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        'health_score':          scan.health_score,
        'condition':             scan.condition,
        'estimated_life_years':  scan.estimated_life_years,
        'critical':              scan.critical,
        'warnings':              scan.warnings_list(),
        'recommendations':       scan.recommendations_list(),
        'material':              scan.material,
        'crack_score':           scan.crack_score,
        'seep_score':            scan.seep_score,
        'surface_score':         scan.surface_score,
        'crack_area_pct':        scan.crack_area_pct,
        'seep_area_pct':         scan.seep_area_pct,
        'num_crack_regions':     scan.num_crack_regions,
        'num_seep_regions':      scan.num_seep_regions,
        'largest_crack_mm_eq':   scan.largest_crack_mm_eq,
        'ai_used':               scan.ai_used,
        'ai_confidence':         round(scan.ai_confidence * 100, 1),
        'annotated_url':         scan.annotated_url,
        'heatmap_url':           scan.heatmap_url,
        'original_url':          scan.original_url,
        'validation_confidence': 100,
    }

    return render(request, 'scanner/result.html', {
        'report':       report,
        'score_class':  score_class,
        'score_pct':    int((scan.health_score / 10) * 100),
        'from_history': True,
    })