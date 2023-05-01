package org.example;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

class ExcelReader {
    private final XSSFWorkbook workbook;

    ExcelReader(String path) throws IOException {
        File file = new File(path);

        FileInputStream fIn = new FileInputStream(file);
        workbook = new XSSFWorkbook(fIn);

        fIn.close();
    }

    public Object[] getRow(int number, int sheetNum) throws Exception {
        XSSFSheet sheet = workbook.getSheetAt(sheetNum);

        int last;
        try{
            last = sheet.getRow(number).getLastCellNum();
        }
        catch (Exception e){
            throw new Exception();
        }

        Object[] data = new Object[last];

        XSSFRow row = sheet.getRow(number);

        for(int i=0;i<last;i++){
            if(row.getCell(i) == null){
                data[i] = null;
                continue;
            }

            if(row.getCell(i).getCellType() == CellType.STRING){
                data[i] = row.getCell(i).getStringCellValue();
            }
            else{
                data[i] = row.getCell(i).getNumericCellValue();
            }
        }

        return data;
    }

    public int getRowCount() {
        return workbook.getSheetAt(0).getLastRowNum() + 1;
    }
}
