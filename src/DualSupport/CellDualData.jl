export CellDualData, update_cell_dual_data_value!, update_cell_dual_data!,
        get_cell_dual, get_m1_dual, get_00_dual, get_p1_dual,
        get_face_00_dual, get_face_p1_dual


"""
    struct CellDualData{NVARSP1, THREENVARSP1, TNUMBER}

Definition of CellDualData, that holds the information needed to construct partial derivatives wrt its own properties
as well as its neighbors.
Parametric in types `NVARSP1`, the number of independent variables plus one, `THREENVARSP1`, three times the number of
independe variables plus one, and `TNUMBER`, the type of the number used for the calculations (usually floats, but
can be duals themselves).
"""
struct CellDualData{NVARSP1, THREENVARSP1, TNUMBER}
    diff_cache_cell::StarDiffCache{NVARSP1, TNUMBER}
    diff_cache_m1::StarDiffCache{THREENVARSP1, TNUMBER}
    diff_cache_00::StarDiffCache{THREENVARSP1, TNUMBER}
    diff_cache_p1::StarDiffCache{THREENVARSP1, TNUMBER}
end

"""
    function CellDualData(nvars::Int, ::Type{TNUMBER}; is_ind_var=false, ind_var_i=0)

Instantiates an object of type CellDualData, that holds the information needed to construct partial derivatives wrt its
own properties as well as its neighbors.
Use `is_ind_var=True` and `ind_var_i=i` to instantiate a CellDualData of a base independent variable,
with ones assigned in the appropriate spots
"""
function CellDualData(nvars::Int, ::Type{TNUMBER}; is_ind_var=false, ind_var_i=0) where{TNUMBER}
    diff_cache_cell = StarDiffCache(nvars, TNUMBER)
    diff_cache_m1 = StarDiffCache(3*nvars, TNUMBER)
    diff_cache_00 = StarDiffCache(3*nvars, TNUMBER)
    diff_cache_p1 = StarDiffCache(3*nvars, TNUMBER)
    cd = CellDualData{nvars+1, 3*nvars+1, TNUMBER}(diff_cache_cell, 
                                diff_cache_m1, diff_cache_00, diff_cache_p1)
    if !is_ind_var
        return cd
    end

    if ind_var_i < 1 || ind_var_i > nvars
        throw(ArgumentError("ind_var_i=$ind_var_i must be larger or equal to one and not greater than nvars=$nvars"))
    end

    # The first entry of dual_du is the value of the variable, rest of entries are partials
    cd.diff_cache_cell.dual_data[1+ind_var_i] = one(TNUMBER)
    # for these, the number of partials is nvars*3, we need to place it accordingly
    cd.diff_cache_m1.dual_data[1+ind_var_i] = one(TNUMBER)
    cd.diff_cache_00.dual_data[1+nvars+ind_var_i] = one(TNUMBER)
    cd.diff_cache_p1.dual_data[1+2*nvars+ind_var_i] = one(TNUMBER)

    return cd
end

"""
    function Base.zero(::Type{CellDualData{SIZE1,SIZE2,TNUMBER}}) where {SIZE1, SIZE2, TNUMBER}

Instantiates a CellDualData with zero entries (the neutral element for duals).
"""
function Base.zero(::Type{CellDualData{SIZE1,SIZE2,TNUMBER}}) where {SIZE1, SIZE2, TNUMBER}
    return CellDualData(SIZE1-1, TNUMBER)
end

"""
    function Base.convert(::Type{CellDualData{SIZE1, SIZE2, TN1}}, x::TN2) where {SIZE1, SIZE2, TN1<:Number, TN2<:Number} 

Convert `x` of type `TN2` to a CellDualData object of types `SIZE1`, `SIZE2` and `TN1`.
"""
function Base.convert(::Type{CellDualData{SIZE1, SIZE2, TN1}}, x::TN2) where {SIZE1, SIZE2, TN1<:Number, TN2<:Number} 
    cd = zero(CellDualData{SIZE1,SIZE2,TN1})
    update_cell_dual_data_value!(cd, x)
    return cd
end

"""
    function update_cell_dual_data_value!(cd::CellDualData, value)

Updates all data of the CellDualData object to the given value.
"""
function update_cell_dual_data_value!(cd::CellDualData, value)
    cd.diff_cache_cell.dual_data[1] = value
    cd.diff_cache_m1.dual_data[1] = value
    cd.diff_cache_00.dual_data[1] = value
    cd.diff_cache_p1.dual_data[1] = value
    # These are some attempts at speeding this up
    #@inbounds cd.diff_cache_cell.dual_data[1] = value
    #@inbounds cd.diff_cache_m1.dual_data[1] = value
    #@inbounds cd.diff_cache_00.dual_data[1] = value
    #@inbounds cd.diff_cache_p1.dual_data[1] = value
end

"""
    function update_cell_dual_data!(cd::CellDualData{SIZE1, SIZE2, TNUMBER}, dual::TDSC) where {SIZE1, SIZE2, TNUMBER, TDSC}

Updates all data of the CellDualData object to the data of a given dual number.
"""
function update_cell_dual_data!(cd::CellDualData{SIZE1, SIZE2, TNUMBER}, dual::TDSC) where {SIZE1, SIZE2, TNUMBER, TDSC}
    update_cell_dual_data_value!(cd, dual.value)
    nvars = SIZE1-1
    for i in 1:nvars
        cd.diff_cache_cell.dual_data[1+i] = dual.partials[i]
        cd.diff_cache_m1.dual_data[1+i] = dual.partials[i]
        cd.diff_cache_00.dual_data[1+nvars+i] = dual.partials[i]
        cd.diff_cache_p1.dual_data[1+2*nvars+i] = dual.partials[i]
    end
    #these are some attempts at speeding this up
    #@inbounds @views cd.diff_cache_cell.dual_data[2:1+nvars] .= dual.partials
    #@inbounds @views cd.diff_cache_m1.dual_data[2:1+nvars] .= dual.partials
    #@inbounds @views cd.diff_cache_00.dual_data[2+nvars:1+2*nvars] .= dual.partials
    #@inbounds @views cd.diff_cache_p1.dual_data[2+2*nvars:1+3*nvars] .= dual.partials
end

function get_value(cd::CellDualData)
    return cd.diff_cache_cell.dual_data[1]
end

function get_cell_dual(cd::CellDualData)
    return get_dual(cd.diff_cache_cell)
end

function get_m1_dual(cd::CellDualData)
    return get_dual(cd.diff_cache_m1)
end

function get_00_dual(cd::CellDualData)
    return get_dual(cd.diff_cache_00)
end

function get_p1_dual(cd::CellDualData)
    return get_dual(cd.diff_cache_p1)
end

function get_face_00_dual(cd::CellDualData)
    return get_face_dual(cd.diff_cache_m1)
end

function get_face_p1_dual(cd::CellDualData)
    return get_face_dual(cd.diff_cache_00)
end
