Attribute VB_Name = "CastRackNames"
Public Function RowRack2Plate(Optional i As String = "")
'converts L24 to D94'
Dim row_l, row_names, out_l As String
Dim row_ix, col_ix, out_n As Integer
If Len(i) = 0 Then
    RowRack2Plate = ""
    Exit Function
End If
row_names = "ABCDEFGHIJKLMNOP"
row_l = UCase(Mid(i, 1, 1))
row_ix = InStr(1, row_names, row_l)
col_ix = CInt(Mid(i, 2))
If row_ix Mod 2 = 1 Then
        If col_ix Mod 2 = 1 Then
            out_l = "A"
        Else:
            out_l = "C"
        End If
    Else
        If col_ix Mod 2 = 1 Then
            out_l = "B"
        Else:
            out_l = "D"
        End If
    End If
    
out_n = 8 * ((col_ix - 1) \ 2) + ((row_ix - 1) \ 2) + 1

RowRack2Plate = out_l & out_n
End Function

Public Function RowRack2PlateLetter(Optional i As String = "")
'converts L24 to D94'
Dim row_l, row_names, out_l As String
Dim row_ix, col_ix, out_n As Integer
If Len(i) = 0 Then
    RowRack2PlateLetter = ""
    Exit Function
End If
row_names = "ABCDEFGHIJKLMNOP"
row_l = UCase(Mid(i, 1, 1))
row_ix = InStr(1, row_names, row_l)
col_ix = CInt(Mid(i, 2))
If row_ix Mod 2 = 1 Then
        If col_ix Mod 2 = 1 Then
            out_l = "A"
        Else:
            out_l = "C"
        End If
    Else
        If col_ix Mod 2 = 1 Then
            out_l = "B"
        Else:
            out_l = "D"
        End If
    End If
    
out_n = 8 * ((col_ix - 1) \ 2) + ((row_ix - 1) \ 2) + 1

RowRack2PlateLetter = out_l
End Function
Public Function RowRack2PlateNumber(Optional i As String = "")
'converts L24 to D94'
Dim row_l, row_names, out_l As String
Dim row_ix, col_ix, out_n As Integer
If Len(i) = 0 Then
    RowRack2PlateNumber = ""
    Exit Function
End If
row_names = "ABCDEFGHIJKLMNOP"
row_l = UCase(Mid(i, 1, 1))
row_ix = InStr(1, row_names, row_l)
col_ix = CInt(Mid(i, 2))
If row_ix Mod 2 = 1 Then
        If col_ix Mod 2 = 1 Then
            out_l = "A"
        Else:
            out_l = "C"
        End If
    Else
        If col_ix Mod 2 = 1 Then
            out_l = "B"
        Else:
            out_l = "D"
        End If
    End If
    
out_n = 8 * ((col_ix - 1) \ 2) + ((row_ix - 1) \ 2) + 1

RowRack2PlateNumber = out_n
End Function
Sub RowRack2PlateRange()
'L24->D94 over range'
'converts a range of L24 in the col A to D in Col E and 94 in Col F'
Dim col_val As String
Range("A2").Select
Do Until False
    col_val = ActiveCell.Value
    If Len(col_val) = 3 Or Len(col_val) = 2 Then
        'goto col d
        ActiveCell.Offset(0, 4).Value = RowRack2PlateLetter(col_val)
        'goto col E
        ActiveCell.Offset(0, 5).Value = RowRack2PlateNumber(col_val)
        ActiveCell.Offset(1, 0).Select
    Else
        Exit Do
    End If
Loop

'sort so when read off the repeats from the cold room everything is in order'
Dim table_rows, table_cols As Integer
Dim whole_table, original_result, plate_num, specimen_rack, tube_num As Range

ActiveCell.Offset(-1, 0).Select
table_rows = ActiveCell.Row
table_cols = 6
Set whole_table = Range(Cells(1, 1), Cells(table_rows, table_cols))
Set original_result = Range(Cells(1, 3), Cells(table_rows, 3))
Set plate_num = Range(Cells(1, 4), Cells(table_rows, 4))
Set specimen_rack = Range(Cells(1, 5), Cells(table_rows, 5))
Set tube_num = Range(Cells(1, 6), Cells(table_rows, 6))

ActiveSheet.Sort.SortFields.Clear
ActiveSheet.Sort.SortFields.Add Key:=original_result _
    , SortOn:=xlSortOnValues, Order:=xlAscending, DataOption:=xlSortNormal
ActiveSheet.Sort.SortFields.Add Key:=plate_num _
    , SortOn:=xlSortOnValues, Order:=xlAscending, DataOption:=xlSortNormal
ActiveSheet.Sort.SortFields.Add Key:=specimen_rack _
    , SortOn:=xlSortOnValues, Order:=xlAscending, DataOption:=xlSortNormal
ActiveSheet.Sort.SortFields.Add Key:=tube_num _
    , SortOn:=xlSortOnValues, Order:=xlAscending, DataOption:=xlSortNormal
With ActiveSheet.Sort
    .SetRange whole_table
    .Header = xlYes 'because the header will be ignored can start from the top row
    .MatchCase = False
    .Orientation = xlTopToBottom
    .SortMethod = xlPinYin
    .Apply
End With

End Sub

