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


def test_normal_booking_message_is_booking_intent():
    result = classify_message("i want to book an appointment")

    assert result.intent == "book_appointment"
    assert result.should_start_booking is True


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

def test_dental_price_question_is_pricing_intent():
    result = classify_message("how much is dental?")

    assert result.intent == "ask_pricing"
    assert result.should_start_booking is False


def test_physiotherapy_price_question_is_pricing_intent():
    result = classify_message("price of physiotherapy")

    assert result.intent == "ask_pricing"
    assert result.should_start_booking is False


def test_service_list_question_is_service_list_intent():
    result = classify_message("what services do you have?")

    assert result.intent == "ask_service_list"
    assert result.should_start_booking is False


def test_available_services_question_is_service_list_intent():
    result = classify_message("show available services")

    assert result.intent == "ask_service_list"
    assert result.should_start_booking is False

def test_opening_hours_question_is_opening_hours_intent():
    result = classify_message("what time are you open?")

    assert result.intent == "ask_opening_hours"
    assert result.should_start_booking is False


def test_closing_time_question_is_opening_hours_intent():
    result = classify_message("when do you close?")

    assert result.intent == "ask_opening_hours"
    assert result.should_start_booking is False


def test_location_question_is_location_intent():
    result = classify_message("where are you located?")

    assert result.intent == "ask_location"
    assert result.should_start_booking is False


def test_address_question_is_location_intent():
    result = classify_message("what is your address?")

    assert result.intent == "ask_location"
    assert result.should_start_booking is False

def test_insurance_question_is_insurance_intent():
    result = classify_message("do you accept insurance?")

    assert result.intent == "ask_insurance"
    assert result.should_start_booking is False


def test_health_insurance_question_is_insurance_intent():
    result = classify_message("can I use health insurance?")

    assert result.intent == "ask_insurance"
    assert result.should_start_booking is False


def test_cancellation_policy_question_is_policy_intent():
    result = classify_message("what is your cancellation policy?")

    assert result.intent == "ask_cancellation_policy"
    assert result.should_start_booking is False


def test_cancel_fee_question_is_policy_intent():
    result = classify_message("is there a cancel fee?")

    assert result.intent == "ask_cancellation_policy"
    assert result.should_start_booking is False


def test_payment_methods_question_is_payment_intent():
    result = classify_message("how can I pay?")

    assert result.intent == "ask_payment_methods"
    assert result.should_start_booking is False


def test_card_payment_question_is_payment_intent():
    result = classify_message("do you accept card payment?")

    assert result.intent == "ask_payment_methods"
    assert result.should_start_booking is False

def test_natural_dentist_need_is_booking_intent():
    result = classify_message("I need a dentist tomorrow")

    assert result.intent == "book_appointment"
    assert result.should_start_booking is True


def test_natural_tooth_doctor_request_is_booking_intent():
    result = classify_message("i need tooth doctor day after tomorrow morning")

    assert result.intent == "book_appointment"
    assert result.should_start_booking is True


def test_natural_consultation_request_is_booking_intent():
    result = classify_message("can I get a consultation next Monday?")

    assert result.intent == "book_appointment"
    assert result.should_start_booking is True


def test_move_my_appointment_is_reschedule_intent():
    result = classify_message("can I move my appointment?")

    assert result.intent == "reschedule_appointment"
    assert result.should_start_booking is False


def test_change_my_slot_is_reschedule_intent():
    result = classify_message("I want to change my slot")

    assert result.intent == "reschedule_appointment"
    assert result.should_start_booking is False


def test_dont_want_my_appointment_is_cancel_intent():
    result = classify_message("I don't want my appointment anymore")

    assert result.intent == "cancel_appointment"
    assert result.should_start_booking is False


def test_cannot_come_is_cancel_intent():
    result = classify_message("I can't come for my appointment")

    assert result.intent == "cancel_appointment"
    assert result.should_start_booking is False