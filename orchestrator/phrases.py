"""Fixed spoken phrases for the deterministic branches (forms, safety, errors), in ar / fr / en.

Booking read-backs come from templates, not the LLM, so what the caller confirms is exactly
what gets saved. Arabic phrases are in simple Algerian Darja, the register callers use.
"""

from __future__ import annotations

from datetime import date, time

from orchestrator.schedule import SCHEDULE, WEEK_ORDER
from orchestrator.state import Lang

PHRASES: dict[str, dict[Lang, str]] = {
    # booking: ask for a missing field
    "ask.department": {
        "fr": "Pour quel service : médecine générale, pédiatrie, cardiologie, gynécologie ou dentisterie ?",
        "en": "Which department: general medicine, pediatrics, cardiology, gynecology or dentistry?",
        "ar": "الموعد في واش من قسم: الطب العام، طب الأطفال، طب القلب، أمراض النساء ولا طب الأسنان؟",
    },
    "ask.appointment_date": {
        "fr": "Quel jour vous conviendrait ?",
        "en": "Which day would suit you?",
        "ar": "واش من نهار يناسبك؟",
    },
    "ask.appointment_time": {
        "fr": "À quelle heure préférez-vous ?",
        "en": "What time would you prefer?",
        "ar": "على واش من ساعة تحب؟",
    },
    "ask.patient_name": {
        "fr": "Quel est le nom complet du patient ?",
        "en": "What is the patient's full name?",
        "ar": "واش هو الاسم الكامل تاع المريض؟",
    },
    "ask.phone": {
        "fr": "Quel est votre numéro de téléphone ?",
        "en": "What is your phone number?",
        "ar": "عطيني رقم التيليفون تاعك من فضلك.",
    },
    # booking: a value was refused
    "unknown_department": {
        "fr": "Ce service ne prend pas de rendez-vous ici. Médecine générale, pédiatrie, cardiologie, gynécologie ou dentisterie ?",
        "en": "We don't book that service. General medicine, pediatrics, cardiology, gynecology or dentistry?",
        "ar": "هاد القسم ما فيهش مواعيد. الطب العام، طب الأطفال، طب القلب، أمراض النساء ولا طب الأسنان؟",
    },
    "invalid_date": {
        "fr": "Je n'ai pas bien compris la date. Quel jour vous conviendrait ?",
        "en": "I didn't catch the date. Which day would suit you?",
        "ar": "ما فهمتش النهار مليح. واش من نهار يناسبك؟",
    },
    "past_date": {
        "fr": "Cette date est déjà passée. Quel autre jour vous conviendrait ?",
        "en": "That date has already passed. Which other day would suit you?",
        "ar": "هاد النهار فات. واش من نهار آخر يناسبك؟",
    },
    "too_far": {
        "fr": "On prend les rendez-vous sur les deux prochains mois seulement. Quel jour plus proche ?",
        "en": "We only book within the next two months. Which earlier day would suit you?",
        "ar": "ناخذو المواعيد غير في الشهرين الجايين. واش من نهار أقرب يناسبك؟",
    },
    "friday_closed": {
        "fr": "La clinique est fermée le vendredi, sauf les urgences. Quel autre jour vous conviendrait ?",
        "en": "The clinic is closed on Fridays except for emergencies. Which other day would suit you?",
        "ar": "العيادة مسكرة نهار الجمعة، غير الاستعجالات اللي تخدم. واش من نهار آخر يناسبك؟",
    },
    "department_closed_that_day": {
        "fr": "En {department}, les consultations ont lieu le {days}. Quel jour vous conviendrait ?",
        "en": "{department} appointments are on {days}. Which day would suit you?",
        "ar": "في {department}، الفحص يكون نهار {days}. واش من نهار يناسبك؟",
    },
    "invalid_time": {
        "fr": "Je n'ai pas bien compris l'heure. À quelle heure préférez-vous ?",
        "en": "I didn't catch the time. What time would you prefer?",
        "ar": "ما فهمتش الساعة مليح. على واش من ساعة تحب؟",
    },
    "outside_hours": {
        "fr": "En {department}, les rendez-vous sont entre {start} et {end}. Quelle heure vous conviendrait ?",
        "en": "{department} appointments are between {start} and {end}. What time would suit you?",
        "ar": "في {department}، المواعيد بين {start} و{end}. واش من ساعة تناسبك؟",
    },
    "slot_taken": {
        "fr": "Ce créneau est déjà pris. Quelle autre heure vous conviendrait ?",
        "en": "That slot is already taken. What other time would suit you?",
        "ar": "هاد الساعة محجوزة. واش من ساعة أخرى تناسبك؟",
    },
    "invalid_name": {
        "fr": "Pouvez-vous me redire le nom complet du patient ?",
        "en": "Could you repeat the patient's full name?",
        "ar": "تقدر تعاودلي الاسم الكامل تاع المريض؟",
    },
    "invalid_phone": {
        "fr": "Je n'ai pas bien saisi le numéro. Pouvez-vous le redire chiffre par chiffre ?",
        "en": "I didn't catch the number. Could you say it again digit by digit?",
        "ar": "ما فهمتش الرقم مليح. تقدر تعاودهولي رقم برقم؟",
    },
    "confirm_booking": {
        "fr": "Je récapitule : {department} le {when}, au nom de {name}, numéro {phone}. C'est bien ça ?",
        "en": "To confirm: {department} on {when}, for {name}, phone {phone}. Is that right?",
        "ar": "نعاودلك: {department} نهار {when}، باسم {name}، الرقم {phone}. صحيح؟",
    },
    "booked": {
        "fr": "C'est noté, votre rendez-vous en {department} est enregistré pour le {when}. Vous recevrez un SMS de confirmation la veille.",
        "en": "Done, your {department} appointment is booked for {when}. You'll get a confirmation SMS the day before.",
        "ar": "صافي، الموعد تاعك في {department} تسجل نهار {when}. يوصلك SMS للتأكيد نهار قبل.",
    },
    "ask_change": {
        "fr": "D'accord, qu'est-ce que je dois corriger ?",
        "en": "Okay, what should I change?",
        "ar": "مليح، واش لازم نبدل؟",
    },
    # callback
    "callback.ask.phone": {
        "fr": "Bien sûr, un membre de l'équipe peut vous rappeler. À quel numéro ?",
        "en": "Of course, a staff member can call you back. On which number?",
        "ar": "أكيد، واحد من الخدامين يعاود يعيطلك. على واش من رقم؟",
    },
    "callback.ask.reason": {
        "fr": "C'est à quel sujet, en quelques mots ?",
        "en": "What is it about, in a few words?",
        "ar": "على واش الموضوع، بكلمتين؟",
    },
    "confirm_callback": {
        "fr": "Je note une demande de rappel au {phone}. C'est bien ça ?",
        "en": "I'll log a callback request to {phone}. Is that right?",
        "ar": "نسجل طلب باش يعيطولك على {phone}. صحيح؟",
    },
    "callback_saved": {
        "fr": "C'est noté, quelqu'un de la clinique vous rappellera dès que possible.",
        "en": "Done, someone from the clinic will call you back as soon as possible.",
        "ar": "صافي، تسجل. واحد من العيادة يعيطلك في أقرب وقت.",
    },
    # safety (never generated by the LLM)
    "emergency": {
        "fr": "Si c'est une urgence, venez tout de suite au service des urgences de la clinique, ouvert 24 heures sur 24, ou appelez le 14.",
        "en": "If this is an emergency, come straight to the clinic's emergency unit, open 24 hours a day, or call 14.",
        "ar": "إذا كانت حالة مستعجلة، روح طول لقسم الاستعجالات تاع العيادة، مفتوح 24 ساعة على 24، ولا عيط على 14.",
    },
    "medical_advice": {
        "fr": "Je ne peux pas donner de conseil médical par téléphone, mais je peux vous prendre un rendez-vous avec un médecin. Vous voulez ?",
        "en": "I can't give medical advice over the phone, but I can book you an appointment with a doctor. Would you like that?",
        "ar": "ما نقدرش نعطي نصائح طبية في التيليفون، بصح نقدر نديرلك موعد مع طبيب. تحب؟",
    },
    # failures
    "tech_error": {
        "fr": "Désolée, j'ai un petit problème technique. Voulez-vous qu'un membre de l'équipe vous rappelle ?",
        "en": "Sorry, I'm having a technical problem. Would you like a staff member to call you back?",
        "ar": "سمحلي، عندي مشكل تقني. تحب واحد من الخدامين يعاود يعيطلك؟",
    },
}

DEPARTMENT_NAMES: dict[Lang, dict[str, str]] = {
    "fr": {
        "general_medicine": "médecine générale",
        "pediatrics": "pédiatrie",
        "cardiology": "cardiologie",
        "gynecology": "gynécologie",
        "dentistry": "dentisterie",
    },
    "en": {
        "general_medicine": "general medicine",
        "pediatrics": "pediatrics",
        "cardiology": "cardiology",
        "gynecology": "gynecology",
        "dentistry": "dentistry",
    },
    "ar": {
        "general_medicine": "الطب العام",
        "pediatrics": "طب الأطفال",
        "cardiology": "طب القلب",
        "gynecology": "أمراض النساء",
        "dentistry": "طب الأسنان",
    },
}

# indexed by datetime.weekday() (Monday = 0)
WEEKDAYS: dict[Lang, tuple[str, ...]] = {
    "fr": ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"),
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
    "ar": ("الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"),
}
MONTHS: dict[Lang, tuple[str, ...]] = {
    "fr": ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
           "septembre", "octobre", "novembre", "décembre"),
    "en": ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"),
    "ar": ("جانفي", "فيفري", "مارس", "أفريل", "ماي", "جوان", "جويلية", "أوت",
           "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"),
}  # fmt: skip
_AND = {"fr": " et ", "en": " and ", "ar": " و"}


def say_time(t: time, lang: Lang) -> str:
    if lang == "fr":
        return f"{t.hour}h{t.minute:02d}" if t.minute else f"{t.hour}h"
    return f"{t.hour}:{t.minute:02d}"


def say_when(d: date, t: time, lang: Lang) -> str:
    day = WEEKDAYS[lang][d.weekday()]
    month = MONTHS[lang][d.month - 1]
    if lang == "fr":
        return f"{day} {d.day} {month} à {say_time(t, lang)}"
    if lang == "en":
        return f"{day}, {month} {d.day} at {say_time(t, lang)}"
    return f"{day} {d.day} {month} على {say_time(t, lang)}"


def say_phone(phone: str) -> str:
    """Grouped for TTS: 0555 12 34 56 / 032 11 22 33."""
    head = 4 if len(phone) == 10 else 3
    rest = phone[head:]
    return " ".join([phone[:head], *(rest[i : i + 2] for i in range(0, len(rest), 2))])


def _join(items: list[str], lang: Lang) -> str:
    sep = "، " if lang == "ar" else ", "
    return items[0] if len(items) == 1 else sep.join(items[:-1]) + _AND[lang] + items[-1]


def say_department(dept: str, lang: Lang) -> str:
    return DEPARTMENT_NAMES[lang].get(dept, dept)


def schedule_ctx(dept: str, lang: Lang) -> dict[str, str]:
    """Placeholders for department_closed_that_day / outside_hours."""
    hours = SCHEDULE[dept]
    days = [WEEKDAYS[lang][d] for d in WEEK_ORDER if d in hours.days]
    return {
        "department": say_department(dept, lang),
        "days": _join(days, lang),
        "start": say_time(hours.start, lang),
        "end": say_time(hours.last_start, lang),
    }


def phrase(key: str, lang: Lang, **ctx: str) -> str:
    text = PHRASES[key][lang]
    if lang == "en" and text.startswith("{department}"):
        ctx = {**ctx, "department": ctx.get("department", "").capitalize()}
    return text.format(**ctx)
