from fiatlux import PropagationRegime, PropagationSamplingError
from fiatlux.optics import PropagationRegime as OpticsPropagationRegime


def test_fraunhofer_is_the_first_and_default_regime():
    assert PropagationRegime.FRAUNHOFER.value == "fraunhofer"
    assert list(PropagationRegime)[0] is PropagationRegime.FRAUNHOFER


def test_fresnel_is_an_independent_physical_regime():
    assert PropagationRegime.FRESNEL.value == "fresnel"


def test_propagation_regime_is_public_and_shared():
    assert OpticsPropagationRegime is PropagationRegime


def test_sampling_error_is_a_value_error():
    assert issubclass(PropagationSamplingError, ValueError)
