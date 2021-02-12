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
Do Until IsEmpty(ActiveCell)
    col_val = ActiveCell.Value
    If Len(col_val) = 3 Or Len(col_val) = 2 Then
        'goto col d
        ActiveCell.Offset(0, 4).Value = RowRack2PlateLetter(col_val)
        'goto col E
        ActiveCell.Offset(0, 5).Value = RowRack2PlateNumber(col_val)
    End If
    ActiveCell.Offset(1, 0).Select
Loop
End Sub

