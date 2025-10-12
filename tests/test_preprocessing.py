"""
Tests unitarios para las funciones de preprocesamiento de datos.
Equipo 59 - Predicción de Ausentismo Laboral
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# ============================================================================
# FIXTURES - Datos de prueba
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Crea un DataFrame de prueba con datos válidos e inválidos."""
    return pd.DataFrame({
        'ID': ['1', '2', '3', '4', '5'],
        'Reason for absence': ['5', '10', '99', '15', '0'],  # 99 es inválido
        'Month of absence': ['1', '6', '15', '12', '0'],     # 15 es inválido
        'Day of the week': ['2', '3', '4', '5', '9'],        # 9 es inválido
        'Seasons': ['1', '2', '3', '4', '1'],
        'Transportation expense': ['100', '200', '300', '1000', '150'],
        'Distance from Residence to Work': ['10', '20', '30', '40', '15'],
        'Service time': ['5', '10', '15', '20', '8'],
        'Age': ['30', '40', '50', '25', '35'],
        'Work load Average/day': ['200', '250', '300', '350', '275'],
        'Hit target': ['90', '85', '95', '80', '88'],
        'Disciplinary failure': ['0', '1', '0', '2', '1'],   # 2 es inválido
        'Education': [1.0, 2.0, 3.0, 4.0, 1.0],
        'Son': ['0', '1', '2', '3', '1'],
        'Social drinker': ['1', '0', '1', '0', '5'],         # 5 es inválido
        'Social smoker': ['0', '1', '0', '1', '0'],
        'Pet': ['0', '1', '2', '3', '1'],
        'Weight': [70.5, 80.2, 90.0, 75.5, 85.0],
        'Height': ['170', '180', '175', '165', '172'],
        'Body mass index': ['24', '25', '29', '27', '28'],
        'Absenteeism time in hours': ['8', '16', '4', '24', '12'],
        'mixed_type_col': ['a', 'b', 'c', 'd', 'e']
    })


@pytest.fixture
def sample_with_nulls():
    """Crea un DataFrame con valores nulos."""
    return pd.DataFrame({
        'ID': ['1', '2', '3'],
        'Reason for absence': ['5', np.nan, '10'],
        'Month of absence': ['1', '6', np.nan],
        'Age': [30, np.nan, 40],
        'Transportation expense': [100, 200, np.nan],
        'Weight': [70.5, np.nan, 80.0],
        'mixed_type_col': ['a', 'b', 'c']
    })


# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO (copiadas del proyecto)
# ============================================================================

def drop_columns(df):
    """Elimina columnas irrelevantes."""
    return df.drop(['ID', 'mixed_type_col'], axis=1, errors='ignore')


def strip_object_columns(df):
    """Elimina espacios en blanco de columnas tipo object."""
    df_obj = df.select_dtypes(include='object').copy()
    for col in df_obj.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def safe_round_to_int_df(df):
    """Convierte valores a enteros de forma segura."""
    def safe_convert(val):
        try:
            if pd.isnull(val):
                return np.nan
            return round(float(val))
        except:
            return np.nan

    for col in df.columns:
        df[col] = df[col].apply(safe_convert)
    return df


def fix_invalid_values(df):
    """Corrige valores inválidos en variables categóricas."""
    df = df.copy()
    
    # Reason for absence: 0-28
    if 'Reason for absence' in df.columns:
        df['Reason for absence'] = df['Reason for absence'].apply(
            lambda x: x if x in range(0, 29) else 0
        )
    
    # Month of absence: 0-12
    if 'Month of absence' in df.columns:
        df['Month of absence'] = df['Month of absence'].apply(
            lambda x: x if x in range(0, 13) else 0
        )
    
    # Day of the week: 2-6
    if 'Day of the week' in df.columns:
        mode_val = df['Day of the week'].mode()[0] if len(df['Day of the week'].mode()) > 0 else 2
        df['Day of the week'] = df['Day of the week'].apply(
            lambda x: x if x in range(2, 7) else mode_val
        )
    
    # Seasons: 1-4
    if 'Seasons' in df.columns:
        mode_val = df['Seasons'].mode()[0] if len(df['Seasons'].mode()) > 0 else 1
        df['Seasons'] = df['Seasons'].apply(
            lambda x: x if x in range(1, 5) else mode_val
        )
    
    # Education: 1-4
    if 'Education' in df.columns:
        mode_val = df['Education'].mode()[0] if len(df['Education'].mode()) > 0 else 1
        df['Education'] = df['Education'].apply(
            lambda x: x if x in range(1, 5) else mode_val
        )
    
    # Binary columns: 0 or 1
    binary_cols = ['Disciplinary failure', 'Social drinker', 'Social smoker']
    for col in binary_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
            df[col] = df[col].apply(lambda x: x if x in [0, 1] else mode_val)
    
    return df


def winsorize_iqr(df):
    """Aplica winsorización IQR para controlar outliers."""
    num_cols = [
        'Transportation expense', 'Distance from Residence to Work', 'Service time',
        'Age', 'Work load Average/day', 'Hit target', 'Son', 'Pet', 'Weight',
        'Height', 'Body mass index', 'Absenteeism time in hours'
    ]
    
    df = df.copy()
    for col in num_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df


def fillna_with_median(df):
    """Rellena valores nulos con la mediana."""
    return df.fillna(df.median(numeric_only=True))


def final_int_conversion(df):
    """Convierte todas las columnas a enteros."""
    return df.round(0).astype(int)


# ============================================================================
# TESTS UNITARIOS
# ============================================================================

class TestDropColumns:
    """Tests para la función drop_columns."""
    
    def test_drops_id_column(self, sample_dataframe):
        """Verifica que se elimine la columna ID."""
        result = drop_columns(sample_dataframe.copy())
        assert 'ID' not in result.columns
    
    def test_drops_mixed_type_col(self, sample_dataframe):
        """Verifica que se elimine la columna mixed_type_col."""
        result = drop_columns(sample_dataframe.copy())
        assert 'mixed_type_col' not in result.columns
    
    def test_preserves_other_columns(self, sample_dataframe):
        """Verifica que se preserven las demás columnas."""
        original_cols = set(sample_dataframe.columns) - {'ID', 'mixed_type_col'}
        result = drop_columns(sample_dataframe.copy())
        assert original_cols == set(result.columns)
    
    def test_row_count_unchanged(self, sample_dataframe):
        """Verifica que no se pierdan filas."""
        result = drop_columns(sample_dataframe.copy())
        assert len(result) == len(sample_dataframe)


class TestStripObjectColumns:
    """Tests para la función strip_object_columns."""
    
    def test_removes_leading_spaces(self):
        """Verifica que se eliminen espacios al inicio."""
        df = pd.DataFrame({'col': [' value', 'value', ' value ']})
        result = strip_object_columns(df.copy())
        assert result['col'].iloc[0] == 'value'
    
    def test_removes_trailing_spaces(self):
        """Verifica que se eliminen espacios al final."""
        df = pd.DataFrame({'col': ['value ', 'value', ' value ']})
        result = strip_object_columns(df.copy())
        assert result['col'].iloc[0] == 'value'
    
    def test_handles_numeric_columns(self, sample_dataframe):
        """Verifica que no falle con columnas numéricas."""
        result = strip_object_columns(sample_dataframe.copy())
        assert result is not None


class TestSafeRoundToInt:
    """Tests para la función safe_round_to_int_df."""
    
    def test_converts_strings_to_int(self):
        """Verifica conversión de strings a enteros."""
        df = pd.DataFrame({'col': ['10.5', '20.7', '30.2']})
        result = safe_round_to_int_df(df.copy())
        assert result['col'].iloc[0] == 11
        assert result['col'].iloc[1] == 21
    
    def test_handles_nan_values(self):
        """Verifica manejo de valores NaN."""
        df = pd.DataFrame({'col': ['10', np.nan, '30']})
        result = safe_round_to_int_df(df.copy())
        assert pd.isna(result['col'].iloc[1])
    
    def test_rounds_correctly(self):
        """Verifica redondeo correcto."""
        df = pd.DataFrame({'col': ['10.4', '10.5', '10.6']})
        result = safe_round_to_int_df(df.copy())
        assert result['col'].iloc[0] == 10
        assert result['col'].iloc[1] == 10  # Python rounds to even
        assert result['col'].iloc[2] == 11


class TestFixInvalidValues:
    """Tests para la función fix_invalid_values."""
    
    def test_fixes_invalid_reason_for_absence(self, sample_dataframe):
        """Verifica corrección de valores inválidos en Reason for absence."""
        df_processed = drop_columns(sample_dataframe.copy())
        df_processed = safe_round_to_int_df(df_processed)
        result = fix_invalid_values(df_processed)
        # El valor 99 debería convertirse a 0
        assert result['Reason for absence'].iloc[2] == 0
    
    def test_fixes_invalid_month(self, sample_dataframe):
        """Verifica corrección de valores inválidos en Month of absence."""
        df_processed = drop_columns(sample_dataframe.copy())
        df_processed = safe_round_to_int_df(df_processed)
        result = fix_invalid_values(df_processed)
        # El valor 15 debería convertirse a 0
        assert result['Month of absence'].iloc[2] == 0
    
    def test_fixes_invalid_binary_values(self, sample_dataframe):
        """Verifica corrección de valores binarios inválidos."""
        df_processed = drop_columns(sample_dataframe.copy())
        df_processed = safe_round_to_int_df(df_processed)
        result = fix_invalid_values(df_processed)
        # Disciplinary failure con valor 2 debe convertirse a moda (0 o 1)
        assert result['Disciplinary failure'].iloc[3] in [0, 1]
    
    def test_preserves_valid_values(self, sample_dataframe):
        """Verifica que los valores válidos no se modifiquen."""
        df_processed = drop_columns(sample_dataframe.copy())
        df_processed = safe_round_to_int_df(df_processed)
        result = fix_invalid_values(df_processed)
        # El primer valor de Reason for absence (5) debe preservarse
        assert result['Reason for absence'].iloc[0] == 5


class TestWinsorizeIQR:
    """Tests para la función winsorize_iqr."""
    
    def test_clips_outliers(self):
        """Verifica que se controlen valores extremos."""
        df = pd.DataFrame({
            'Age': [20, 25, 30, 35, 40, 100],  # 100 es outlier
            'Weight': [60, 65, 70, 75, 80, 85]
        })
        result = winsorize_iqr(df.copy())
        # El valor 100 debería ser reducido
        assert result['Age'].max() < 100
    
    def test_preserves_normal_values(self):
        """Verifica que valores normales se preserven."""
        df = pd.DataFrame({
            'Age': [25, 30, 35, 40],
            'Weight': [65, 70, 75, 80]
        })
        result = winsorize_iqr(df.copy())
        # Los valores deben permanecer relativamente iguales
        assert result['Age'].mean() == pytest.approx(df['Age'].mean(), rel=0.1)
    
    def test_handles_missing_columns(self):
        """Verifica que no falle con columnas faltantes."""
        df = pd.DataFrame({'Other_Column': [1, 2, 3, 4, 5]})
        result = winsorize_iqr(df.copy())
        assert 'Other_Column' in result.columns


class TestFillnaWithMedian:
    """Tests para la función fillna_with_median."""
    
    def test_fills_nan_with_median(self):
        """Verifica que los NaN se rellenen con la mediana."""
        df = pd.DataFrame({'col': [10, 20, np.nan, 40, 50]})
        result = fillna_with_median(df.copy())
        # Mediana de [10, 20, 40, 50] = 30
        assert result['col'].iloc[2] == 30.0
    
    def test_preserves_non_nan_values(self):
        """Verifica que valores no-NaN se preserven."""
        df = pd.DataFrame({'col': [10, 20, 30, 40, 50]})
        result = fillna_with_median(df.copy())
        assert (result['col'] == df['col']).all()
    
    def test_no_nan_after_filling(self, sample_with_nulls):
        """Verifica que no queden NaN después del proceso."""
        df_processed = drop_columns(sample_with_nulls.copy())
        df_processed = safe_round_to_int_df(df_processed)
        result = fillna_with_median(df_processed)
        # Solo debe haber NaN en columnas no numéricas
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isna().sum().sum() == 0


class TestFinalIntConversion:
    """Tests para la función final_int_conversion."""
    
    def test_converts_to_int(self):
        """Verifica conversión a enteros."""
        df = pd.DataFrame({'col': [10.5, 20.7, 30.2]})
        result = final_int_conversion(df.copy())
        assert result['col'].dtype == 'int64'
    
    def test_rounds_before_conversion(self):
        """Verifica redondeo antes de conversión."""
        df = pd.DataFrame({'col': [10.4, 10.6, 20.5]})
        result = final_int_conversion(df.copy())
        assert result['col'].iloc[0] == 10
        assert result['col'].iloc[1] == 11


class TestFullPipeline:
    """Tests de integración para el pipeline completo."""
    
    def test_pipeline_runs_without_error(self, sample_dataframe):
        """Verifica que el pipeline completo se ejecute sin errores."""
        preprocessing_pipeline = Pipeline([
            ('drop_columns', FunctionTransformer(drop_columns)),
            ('strip_objects', FunctionTransformer(strip_object_columns)),
            ('safe_round', FunctionTransformer(safe_round_to_int_df)),
            ('fix_invalids', FunctionTransformer(fix_invalid_values)),
            ('winsorize', FunctionTransformer(winsorize_iqr)),
            ('fillna', FunctionTransformer(fillna_with_median)),
            ('final_int', FunctionTransformer(final_int_conversion))
        ])
        
        result = preprocessing_pipeline.fit_transform(sample_dataframe.copy())
        assert result is not None
    
    def test_pipeline_removes_nulls(self, sample_with_nulls):
        """Verifica que el pipeline elimine todos los valores nulos."""
        preprocessing_pipeline = Pipeline([
            ('drop_columns', FunctionTransformer(drop_columns)),
            ('strip_objects', FunctionTransformer(strip_object_columns)),
            ('safe_round', FunctionTransformer(safe_round_to_int_df)),
            ('fix_invalids', FunctionTransformer(fix_invalid_values)),
            ('winsorize', FunctionTransformer(winsorize_iqr)),
            ('fillna', FunctionTransformer(fillna_with_median)),
            ('final_int', FunctionTransformer(final_int_conversion))
        ])
        
        result = preprocessing_pipeline.fit_transform(sample_with_nulls.copy())
        assert result.isna().sum().sum() == 0
    
    def test_pipeline_output_shape(self, sample_dataframe):
        """Verifica que el pipeline mantenga el número de filas."""
        preprocessing_pipeline = Pipeline([
            ('drop_columns', FunctionTransformer(drop_columns)),
            ('strip_objects', FunctionTransformer(strip_object_columns)),
            ('safe_round', FunctionTransformer(safe_round_to_int_df)),
            ('fix_invalids', FunctionTransformer(fix_invalid_values)),
            ('winsorize', FunctionTransformer(winsorize_iqr)),
            ('fillna', FunctionTransformer(fillna_with_median)),
            ('final_int', FunctionTransformer(final_int_conversion))
        ])
        
        result = preprocessing_pipeline.fit_transform(sample_dataframe.copy())
        assert len(result) == len(sample_dataframe)
    
    def test_pipeline_output_types(self, sample_dataframe):
        """Verifica que todas las columnas sean de tipo entero."""
        preprocessing_pipeline = Pipeline([
            ('drop_columns', FunctionTransformer(drop_columns)),
            ('strip_objects', FunctionTransformer(strip_object_columns)),
            ('safe_round', FunctionTransformer(safe_round_to_int_df)),
            ('fix_invalids', FunctionTransformer(fix_invalid_values)),
            ('winsorize', FunctionTransformer(winsorize_iqr)),
            ('fillna', FunctionTransformer(fillna_with_median)),
            ('final_int', FunctionTransformer(final_int_conversion))
        ])
        
        result = preprocessing_pipeline.fit_transform(sample_dataframe.copy())
        assert all(result.dtypes == 'int64')


class TestDataValidation:
    """Tests de validación de datos."""
    
    def test_reason_for_absence_in_valid_range(self, sample_dataframe):
        """Verifica que Reason for absence esté en rango válido."""
        preprocessing_pipeline = Pipeline([
            ('drop_columns', FunctionTransformer(drop_columns)),
            ('strip_objects', FunctionTransformer(strip_object_columns)),
            ('safe_round', FunctionTransformer(safe_round_to_int_df)),
            ('fix_invalids', FunctionTransformer(fix_invalid_values)),
        ])
        
        result = preprocessing_pipeline.fit_transform(sample_dataframe.copy())
        assert result['Reason for absence'].between(0, 28).all()
    
    def test_month_in_valid_range(self, sample_dataframe):
        """Verifica que Month of absence esté en rango válido."""
        preprocessing_pipeline = Pipeline([
            ('drop_columns', FunctionTransformer(drop_columns)),
            ('strip_objects', FunctionTransformer(strip_object_columns)),
            ('safe_round', FunctionTransformer(safe_round_to_int_df)),
            ('fix_invalids', FunctionTransformer(fix_invalid_values)),
        ])
        
        result = preprocessing_pipeline.fit_transform(sample_dataframe.copy())
        assert result['Month of absence'].between(0, 12).all()
    
    def test_binary_columns_are_binary(self, sample_dataframe):
        """Verifica que las columnas binarias solo contengan 0 o 1."""
        preprocessing_pipeline = Pipeline([
            ('drop_columns', FunctionTransformer(drop_columns)),
            ('strip_objects', FunctionTransformer(strip_object_columns)),
            ('safe_round', FunctionTransformer(safe_round_to_int_df)),
            ('fix_invalids', FunctionTransformer(fix_invalid_values)),
            ('winsorize', FunctionTransformer(winsorize_iqr)),
            ('fillna', FunctionTransformer(fillna_with_median)),
            ('final_int', FunctionTransformer(final_int_conversion))
        ])
        
        result = preprocessing_pipeline.fit_transform(sample_dataframe.copy())
        
        binary_cols = ['Disciplinary failure', 'Social drinker', 'Social smoker']
        for col in binary_cols:
            if col in result.columns:
                assert result[col].isin([0, 1]).all(), f"{col} contains non-binary values"

