from agent.llm.schemas.covenants import CovenantExtract

def test_springing_type_is_stable():
    """Аннотация и реальный класс должны совпадать между вызовами."""
    a = CovenantExtract.model_fields["springing"].annotation
    b = CovenantExtract.model_fields["springing"].annotation
    assert a is b
    assert "<locals>" not in str(a)