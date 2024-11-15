# test_arrays_pytest.py

import pytest
from arrays import (
    Array,
    ArrayAlgorithms,
    MultiDimensionalArray,
    SparseArray,
    ArrayUtilities
)


@pytest.fixture
def empty_array():
    return Array[int]()


@pytest.fixture
def populated_array():
    arr = Array[int]()
    for i in [1, 3, 5, 7, 9]:
        arr.append(i)
    return arr


def test_array_initialization(empty_array):
    assert len(empty_array) == 0
    assert empty_array._capacity == 16


def test_array_append(empty_array):
    empty_array.append(10)
    assert len(empty_array) == 1
    assert empty_array[0] == 10


def test_array_get_set(empty_array):
    empty_array.append(20)
    assert empty_array[0] == 20
    empty_array[0] = 30
    assert empty_array[0] == 30
    with pytest.raises(IndexError):
        _ = empty_array[1]


def test_array_insert(populated_array):
    populated_array.insert(2, 4)
    assert populated_array.to_list() == [1, 3, 4, 5, 7, 9]
    with pytest.raises(IndexError):
        populated_array.insert(10, 100)


def test_array_remove(populated_array):
    populated_array.remove(3)
    assert populated_array.to_list() == [1, 5, 7, 9]
    with pytest.raises(ValueError):
        populated_array.remove(100)


def test_array_pop(populated_array):
    popped = populated_array.pop(1)
    assert popped == 3
    assert len(populated_array) == 4
    with pytest.raises(IndexError):
        populated_array.pop(10)


def test_array_index_of(populated_array):
    assert populated_array.index_of(5) == 2
    assert populated_array.index_of(100) == -1


def test_array_clear(populated_array):
    populated_array.clear()
    assert len(populated_array) == 0


def test_array_contains(populated_array):
    assert populated_array.contains(3)
    assert not populated_array.contains(100)


def test_array_resize():
    array = Array[int](2)
    array.append(1)
    array.append(2)
    array.append(3)  # Triggers resize
    assert array._capacity == 4
    for _ in range(3):
        array.pop(0)
    assert array._capacity == 2


def test_linear_search(populated_array):
    index = ArrayAlgorithms.linear_search(populated_array, 5)
    assert index == 2
    index = ArrayAlgorithms.linear_search(populated_array, 100)
    assert index == -1


def test_binary_search(populated_array):
    index = ArrayAlgorithms.binary_search(populated_array, 7)
    assert index == 3
    index = ArrayAlgorithms.binary_search(populated_array, 4)
    assert index == -1


def test_bubble_sort():
    unsorted = Array[int]()
    for num in [5, 1, 4, 2, 8]:
        unsorted.append(num)
    ArrayAlgorithms.bubble_sort(unsorted)
    assert unsorted.to_list() == [1, 2, 4, 5, 8]


def test_selection_sort():
    unsorted = Array[int]()
    for num in [64, 25, 12, 22, 11]:
        unsorted.append(num)
    ArrayAlgorithms.selection_sort(unsorted)
    assert unsorted.to_list() == [11, 12, 22, 25, 64]


def test_insertion_sort():
    unsorted = Array[int]()
    for num in [12, 11, 13, 5, 6]:
        unsorted.append(num)
    ArrayAlgorithms.insertion_sort(unsorted)
    assert unsorted.to_list() == [5, 6, 11, 12, 13]


def test_quick_sort():
    unsorted = Array[int]()
    for num in [10, 7, 8, 9, 1, 5]:
        unsorted.append(num)
    ArrayAlgorithms.quick_sort(unsorted)
    assert unsorted.to_list() == [1, 5, 7, 8, 9, 10]


def test_merge_sort():
    unsorted = Array[int]()
    for num in [38, 27, 43, 3, 9, 82, 10]:
        unsorted.append(num)
    ArrayAlgorithms.merge_sort(unsorted)
    assert unsorted.to_list() == [3, 9, 10, 27, 38, 43, 82]


@pytest.fixture
def multi_dimensional_array():
    return MultiDimensionalArray([2, 3])


def test_multi_dimensional_initialization(multi_dimensional_array):
    assert multi_dimensional_array.dimensions == [2, 3]
    assert multi_dimensional_array.size == 6


def test_multi_dimensional_set_get(multi_dimensional_array):
    multi_dimensional_array.set([0, 0], 'a')
    multi_dimensional_array.set([1, 2], 'b')
    assert multi_dimensional_array.get([0, 0]) == 'a'
    assert multi_dimensional_array.get([1, 2]) == 'b'
    with pytest.raises(IndexError):
        multi_dimensional_array.get([2, 0])


def test_multi_dimensional_to_nested_list(multi_dimensional_array):
    values = ['a', 'b', 'c', 'd', 'e', 'f']
    for i in range(2):
        for j in range(3):
            multi_dimensional_array.set([i, j], values[i * 3 + j])
    nested = multi_dimensional_array.to_nested_list()
    assert nested == [['a', 'b', 'c'], ['d', 'e', 'f']]


@pytest.fixture
def sparse_array():
    return SparseArray[int](5)


def test_sparse_initialization(sparse_array):
    assert sparse_array.size == 5
    assert sparse_array.to_dense() == [0, 0, 0, 0, 0]


def test_sparse_set_get(sparse_array):
    sparse_array[0] = 10
    sparse_array[3] = 20
    assert sparse_array[0] == 10
    assert sparse_array[3] == 20
    assert sparse_array[1] == 0
    with pytest.raises(IndexError):
        _ = sparse_array[5]


def test_sparse_set_zero(sparse_array):
    sparse_array[2] = 30
    assert sparse_array[2] == 30
    sparse_array[2] = 0
    assert sparse_array[2] == 0
    assert 2 not in sparse_array.data


def test_sparse_to_dense(sparse_array):
    sparse_array[1] = 15
    sparse_array[4] = 25
    assert sparse_array.to_dense() == [0, 15, 0, 0, 25]


def test_sparse_non_zero_elements(sparse_array):
    sparse_array[0] = 5
    sparse_array[2] = 10
    assert sorted(sparse_array.non_zero_elements()) == [(0, 5), (2, 10)]


@pytest.fixture
def array_utilities_array():
    arr = Array[int]()
    for num in [1, 2, 3, 2, 4, 3, 5]:
        arr.append(num)
    return arr


def test_find_duplicates(array_utilities_array):
    duplicates = ArrayUtilities.find_duplicates(array_utilities_array)
    assert duplicates.to_list() == [2, 3]


def test_rotate_right(array_utilities_array):
    ArrayUtilities.rotate_right(array_utilities_array, 2)
    assert array_utilities_array.to_list() == [4, 3, 1, 2, 3, 2, 5]


def test_rotate_right_full_rotation(array_utilities_array):
    original = array_utilities_array.to_list().copy()
    ArrayUtilities.rotate_right(array_utilities_array, len(array_utilities_array))
    assert array_utilities_array.to_list() == original


def test_rotate_right_zero(array_utilities_array):
    original = array_utilities_array.to_list().copy()
    ArrayUtilities.rotate_right(array_utilities_array, 0)
    assert array_utilities_array.to_list() == original


def test_rotate_right_negative(array_utilities_array):
    with pytest.raises(ValueError):
        ArrayUtilities.rotate_right(array_utilities_array, -1)


def test_maximum_subarray():
    arr = Array[int]()
    for num in [-2,1,-3,4,-1,2,1,-5,4]:
        arr.append(num)
    max_sum = ArrayUtilities.maximum_subarray(arr)
    assert max_sum == 6  # [4,-1,2,1]


def test_maximum_subarray_empty():
    empty_arr = Array[int]()
    with pytest.raises(ValueError):
        ArrayUtilities.maximum_subarray(empty_arr)


def test_merge_two_sorted_arrays():
    arr1 = Array[int]()
    arr2 = Array[int]()
    for num in [1, 3, 5]:
        arr1.append(num)
    for num in [2, 4, 6]:
        arr2.append(num)
    merged = ArrayUtilities.merge_two_sorted_arrays(arr1, arr2)
    assert merged.to_list() == [1, 2, 3, 4, 5, 6]


def test_merge_two_sorted_arrays_with_duplicates():
    arr1 = Array[int]()
    arr2 = Array[int]()
    for num in [1, 2, 2, 3]:
        arr1.append(num)
    for num in [2, 3, 4]:
        arr2.append(num)
    merged = ArrayUtilities.merge_two_sorted_arrays(arr1, arr2)
    print(merged.to_list() == [1, 2, 2, 2, 3, 3, 4])
