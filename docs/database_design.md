# Core Database Relationships

## Service and Staff

A staff member can provide multiple services, and a service can be
provided by multiple staff members. This is a many-to-many relationship
implemented using the StaffService association table.

## Customer and Appointment

A customer can have multiple appointments. Each appointment belongs to
one customer.

## Staff and Availability

A staff member can have multiple availability slots. Each slot belongs
to one staff member and one service.

## Appointment Relationships

Each appointment belongs to one customer, one service, one staff member
and one availability slot.

## Availability Strategy

The first version uses pre-generated availability slots because this
simplifies availability checks, booking validation and double-booking
prevention.