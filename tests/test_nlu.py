from app.ai.nlu import classify_message


def test_notification_question_does_not_start_booking():
    result = classify_message("if i book an appointment will you notify me")

    assert result.intent == "ask_notification_capability"
    assert result.should_start_booking is False


def test_notify_me_is_notification_question():
    result = classify_message("notify me please")

    assert result.intent == "ask_notification_capability"
    assert result.should_start_booking is False


def test_sms_reminder_is_notification_question():
    result = classify_message("will I get an SMS reminder")

    assert result.intent == "ask_notification_capability"
    assert result.should_start_booking is False


def test_blank_message():
    result = classify_message("   ")

    assert result.intent == "blank"


def test_normal_booking_message_is_not_notification_question():
    result = classify_message("i want to book an appointment")

    assert result.intent == "unknown"


def test_dental_question_is_service_availability_question():
    result = classify_message("do you have dental?")

    assert result.intent == "ask_service_availability"
    assert result.should_start_booking is False


def test_physiotherapy_question_is_service_availability_question():
    result = classify_message("do you offer physiotherapy?")

    assert result.intent == "ask_service_availability"
    assert result.should_start_booking is False


def test_surgery_question_is_service_availability_question():
    result = classify_message("do you do surgery?")

    assert result.intent == "ask_service_availability"
    assert result.should_start_booking is False


def test_tooth_cleaning_question_is_service_availability_question():
    result = classify_message("do you do tooth cleaning?")

    assert result.intent == "ask_service_availability"
    assert result.should_start_booking is False