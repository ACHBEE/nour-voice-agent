# 2-minute demo script

Goal: show, in one take, a French call, a Darja question, an interruption, a booking saved to
the database, and an out-of-scope question handled honestly. Every line below was chosen so the
answer is in the clinic documents (or deliberately is not).

## Setup (before recording)

1. Start the stack: `docker compose up -d postgres` and `python -m rag_service.server`
   (or the full `docker compose up -d --build` and the LiveKit Agents Playground).
2. In a second terminal: `python agent/agent.py console` (add `--input-device N` if needed).
3. Screen layout: agent console on the left; on the right a terminal ready to run
   `docker exec nour-postgres psql -U nour -d nour -c "SELECT patient_name, phone, department, appointment_date, appointment_time, language FROM bookings ORDER BY id DESC LIMIT 1"`.
4. Pick a booking day that is a **Sunday to Thursday in the future** (dentistry is closed on
   Fridays) and a name and number to say.

## Script

| Time | You say | Nour should | Point to on screen |
|---|---|---|---|
| 0:00-0:10 | (nothing; show the README diagram, then start the call) | Greet in French and Arabic | "LiveKit voice agent, LangGraph orchestrator, RAG over gRPC" |
| 0:10-0:25 | **FR:** "Bonjour, combien coûte une échographie abdominale ?" | "3000 dinars" (maybe after "Un instant, je vérifie.") | The filler is deliberate: it covers retrieval on slow turns |
| 0:25-0:45 | **Darja:** "واش تقبلو كارط الشفاء؟" | Yes, for general medicine and lab tests, you pay 20% (in Darja) | The answer comes from an Arabic document; the question was Darja |
| 0:45-1:00 | **FR:** "Quels sont les horaires pendant le Ramadan ?" then, while Nour is still talking: "Attendez, et le laboratoire ?" | Stop mid-sentence, then give lab hours (Ramadan: 08:30 to 11:00) | Barge-in: interruption handled by LiveKit, graph turn cancelled |
| 1:00-1:40 | **Darja:** "نحب ندير رونديفو عند طبيب الأسنان نهار الأحد على العشرة" then the name, then the phone number digit by digit, then "إيه صحيح" | Ask for the name, then the number, read everything back ("صحيح؟"), then confirm the booking and mention the SMS | Run the psql command: the row appears with `language = ar` |
| 1:40-1:55 | **EN:** "Do you have wifi in the waiting room?" | Say she doesn't have that information and offer a staff callback | Not in any document: Nour doesn't invent a yes or no (an eval caught that) |
| 1:55-2:00 | (end the call) | | README results table: 95% answer correctness, 0 medical-advice replies |

## Backup lines

If something goes wrong on camera, these are reliable alternatives:

- Emergency: "J'ai une douleur dans la poitrine" -> fixed safety phrase (emergency unit or 14).
- Medical advice: "What medicine can I take for a headache?" -> refuses, offers an appointment.
- Rule check during booking: ask for pediatrics on a Tuesday -> Nour gives the real days
  (Saturday, Monday, Wednesday).
