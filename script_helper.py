from cProfile import run
import wx
import wx.lib.imagebrowser as ib
import os

"""
Simple GUI to make it easier to run style_tranfer.py without having to do it through the console.
"""
class ScriptHelperFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(ScriptHelperFrame, self).__init__(*args, **kw)
        # create a panel in the frame
        self.pnl = wx.Panel(self)

        # Dictionary storing args to give to script
        self.script_args = {
            "output_image_path" : "\"C:/Users/Patrick/Desktop/Anaconda Files/Style Transfer/Output Images/stylized.jpg\""
        }

        # Dictionary relating ids to object names to be used in the script call
        self.names = {
            -31976 : "style_content_ratio",
            -31974 : "min_improve",
            -31972 : "max_epochs",
            -31970 : "variation_weight",
            -31968 : "preserve_color",
            -31966 : "content_layer",
            -31964 : "pooling",
            -31962 : "output_image_path",
            -31983 : "style_image_path",
            -31982 : "content_image_path",
            -31960 : "max_size",
            -31958 : "style_layer",
            -31956 : "mask_path",
            -31952 : "upscale",
            -31950 : "start_image",
            -31954 : "invert_mask"
        }

        # Title text
        title = wx.StaticText(self.pnl, label="Script Helper for Style Transfer")
        font = title.GetFont()
        font.PointSize += 3
        font = font.Bold()
        title.SetFont(font)
        params = wx.StaticText(self.pnl, label="Parameters")
        params.SetFont(font)

        # Style and content image selection buttons
        bmp = wx.Bitmap("./res/image icon small.png", wx.BITMAP_TYPE_ANY)
        self.style_image_button = wx.BitmapButton(self.pnl, bitmap=bmp)
        self.Bind(wx.EVT_BUTTON, self.OnImageSelect, self.style_image_button, id=1)
        self.content_image_button = wx.BitmapButton(self.pnl, bitmap=bmp)
        self.Bind(wx.EVT_BUTTON, self.OnImageSelect, self.content_image_button, id=2)

        # Labels for style/content image selection buttons
        style_label = wx.StaticText(self.pnl, label="Style Image")
        content_label = wx.StaticText(self.pnl, label="Content Image")

        # View/run buttons
        view_button = wx.Button(self.pnl, label="View")
        self.Bind(wx.EVT_BUTTON, self.OnView, view_button)
        run_button = wx.Button(self.pnl, label="Run")
        self.Bind(wx.EVT_BUTTON, self.OnRun, run_button)

        ### Parameter entry fields
        content_layer_choices = ["block2_conv2", "block3_conv2", "block4_conv2", "block5_conv2"]
        style_layer_choices = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1',
                               'block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2', 'block5_conv2',
                               'block3_conv3', 'block4_conv3', 'block5_conv3',
                               'block3_conv4', 'block4_conv4', 'block5_conv4']

        ratio_label = wx.StaticText(self.pnl, label="Content/Style Ratio")
        ratio_field = wx.TextCtrl(self.pnl, value="0.001")
        self.Bind(wx.EVT_TEXT, self.OnTextUpdate, ratio_field)
        min_improve_label = wx.StaticText(self.pnl, label="Min improve")
        min_improve_field = wx.TextCtrl(self.pnl, value="100")
        self.Bind(wx.EVT_TEXT, self.OnTextUpdate, min_improve_field)
        max_epochs_label = wx.StaticText(self.pnl, label="Max epochs")
        max_epochs_field = wx.TextCtrl(self.pnl, value="100")
        self.Bind(wx.EVT_TEXT, self.OnTextUpdate, max_epochs_field)
        var_weight_label = wx.StaticText(self.pnl, label="Variation weight")
        var_weight_field = wx.TextCtrl(self.pnl, value="0.0001")
        self.Bind(wx.EVT_TEXT, self.OnTextUpdate, var_weight_field)
        preserve_color_label = wx.StaticText(self.pnl, label="Preserve content color")
        preserve_color_field = wx.CheckBox(self.pnl)
        self.Bind(wx.EVT_CHECKBOX, self.OnCheck, preserve_color_field)
        content_layer_label = wx.StaticText(self.pnl, label="Content layer(s)")
        content_layer_field = wx.CheckListBox(self.pnl, style=wx.LB_MULTIPLE | wx.LB_NEEDED_SB, size=(120, 80), choices=content_layer_choices)
        self.Bind(wx.EVT_CHECKLISTBOX, self.OnMultiSelect, content_layer_field)
        self.Bind(wx.EVT_LISTBOX, self.UnSelectAll, content_layer_field)
        pooling_label = wx.StaticText(self.pnl, label="Pooling type")
        pooling_field = wx.ComboBox(self.pnl, style=wx.CB_DROPDOWN | wx.CB_READONLY, value="avg", choices=["avg", "max"])
        self.Bind(wx.EVT_COMBOBOX, self.OnDropdownSelect, pooling_field)
        output_file_label = wx.StaticText(self.pnl, label="Output file name")
        output_file_field = wx.TextCtrl(self.pnl, value="stylized.jpg")
        self.Bind(wx.EVT_TEXT, self.OnTextUpdate, output_file_field)
        max_size_label = wx.StaticText(self.pnl, label="Max image size")
        max_size_field = wx.TextCtrl(self.pnl, value="500")
        self.Bind(wx.EVT_TEXT, self.OnTextUpdate, max_size_field)
        style_layer_label = wx.StaticText(self.pnl, label="Style layer(s)")
        style_layer_field = wx.CheckListBox(self.pnl, style=wx.LB_MULTIPLE | wx.LB_NEEDED_SB, size=(120, 80), choices=style_layer_choices)
        self.Bind(wx.EVT_CHECKLISTBOX, self.OnMultiSelect, style_layer_field)
        self.Bind(wx.EVT_LISTBOX, self.UnSelectAll, style_layer_field)

        # Mask selection button
        mask_label = wx.StaticText(self.pnl, label="Mask image")
        self.mask_button = wx.BitmapButton(self.pnl, bitmap=bmp)
        self.Bind(wx.EVT_BUTTON, self.OnImageSelect, self.mask_button)

        # Mask invert button
        mask_invert_label = wx.StaticText(self.pnl, label="Invert Mask")
        mask_invert_field = wx.CheckBox(self.pnl)
        self.Bind(wx.EVT_CHECKBOX, self.OnCheck, mask_invert_field)

        # Upscaling button
        upscale_label = wx.StaticText(self.pnl, label="Upscale output")
        upscale_field = wx.CheckBox(self.pnl)
        self.Bind(wx.EVT_CHECKBOX, self.OnCheck, upscale_field)
        
        # Starting point dropdown select
        starting_point_label = wx.StaticText(self.pnl, label="Start image")
        starting_point_field = wx.ComboBox(self.pnl, style=wx.CB_DROPDOWN | wx.CB_READONLY, value="content", choices=["content", "style", "white_noise"])
        self.Bind(wx.EVT_COMBOBOX, self.OnDropdownSelect, starting_point_field)

        ### Sizers for organizing page layout
        # Main title and parameters title 
        title_sizer = wx.BoxSizer(wx.HORIZONTAL)
        title_sizer.Add(title, 0, wx.TOP | wx.LEFT)
        title_sizer.Add(params, 0, wx.LEFT, 25)

        # Grid of parameter inputs
        param_grid_sizer = wx.GridSizer(6, 4, -20, 5)
        param_grid_sizer.Add(upscale_label)
        param_grid_sizer.Add(upscale_field)
        param_grid_sizer.Add(preserve_color_label)
        param_grid_sizer.Add(preserve_color_field)
        param_grid_sizer.Add(ratio_label)
        param_grid_sizer.Add(ratio_field)
        param_grid_sizer.Add(max_epochs_label)
        param_grid_sizer.Add(max_epochs_field)
        param_grid_sizer.Add(min_improve_label)
        param_grid_sizer.Add(min_improve_field)
        param_grid_sizer.Add(starting_point_label)
        param_grid_sizer.Add(starting_point_field)
        param_grid_sizer.Add(var_weight_label)
        param_grid_sizer.Add(var_weight_field)
        param_grid_sizer.Add(pooling_label)
        param_grid_sizer.Add(pooling_field)
        param_grid_sizer.Add(max_size_label)
        param_grid_sizer.Add(max_size_field)
        param_grid_sizer.Add(output_file_label)
        param_grid_sizer.Add(output_file_field)
        param_grid_sizer.Add(content_layer_label)
        param_grid_sizer.Add(content_layer_field)
        param_grid_sizer.Add(style_layer_label)
        param_grid_sizer.Add(style_layer_field)

        # Invert mask sizer (to go inside image input sizer)
        invert_sizer = wx.BoxSizer(wx.HORIZONTAL)
        invert_sizer.Add(mask_invert_label, 0, wx.LEFT | wx.TOP, 5)
        invert_sizer.Add(mask_invert_field, 0, wx.LEFT | wx.TOP, 5)

        # Image inputs
        image_sizer = wx.BoxSizer(wx.VERTICAL)
        image_sizer.Add(style_label, 0, wx.LEFT | wx.TOP, 5)
        image_sizer.Add(self.style_image_button, 0, wx.TOP | wx.ALIGN_CENTER_HORIZONTAL, 5)
        image_sizer.Add(content_label, 0, wx.LEFT | wx.TOP, 5)
        image_sizer.Add(self.content_image_button, 0, wx.TOP | wx.ALIGN_CENTER_HORIZONTAL, 5)
        image_sizer.Add(mask_label, 0, wx.LEFT | wx.TOP, 5)
        image_sizer.Add(self.mask_button, 0, wx.TOP | wx.ALIGN_CENTER_HORIZONTAL, 5)
        image_sizer.Add(invert_sizer, 0, wx.TOP , 0, 5)

        # Inner sizer
        inner_sizer = wx.BoxSizer(wx.HORIZONTAL)
        inner_sizer.Add(image_sizer)
        inner_sizer.Add(param_grid_sizer, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 150)

        # Bottom sizer
        bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        bottom_sizer.Add(view_button, 0, wx.RIGHT, 20)
        bottom_sizer.Add(run_button, 0, wx.RIGHT, 20)

        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(title_sizer)
        main_sizer.Add(inner_sizer)
        main_sizer.Add(bottom_sizer, 0, wx.ALIGN_RIGHT | wx.BOTTOM | wx.TOP, 20)
        self.pnl.SetSizer(main_sizer)

        # create a menu bar
        self.makeMenuBar()

    def makeMenuBar(self):
        # Help menu
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)
        menuBar = wx.MenuBar()
        menuBar.Append(helpMenu, "&Help")

        # Apply menu bar to the frame
        self.SetMenuBar(menuBar)

        # Show about information (describing parameters)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)

    def OnAbout(self, event):
        """Display information about what each of the parameters does and how to use the script helper"""
        wx.MessageBox("This popup will eventually have real information")
    
    def OnImageSelect(self, event):
        if event.GetId() == -31982:
            path = "./Content Images"
        elif event.GetId() == -31983:
            path = "./Style Images"
        else:
            path = "./Mask Images"
        with ib.ImageDialog(None, set_dir=path) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                # check which button was selected
                selected = event.GetEventObject()
                # update image button to display selected image and add label with file path
                file_path = dlg.GetFile()
                new_bmp = wx.Bitmap(wx.Image(file_path).Rescale(100,86,3))
                selected.SetBitmap(new_bmp)
                # save selected image's file path to args dictionary
                self.script_args[self.names[event.GetId()]] = "\"" + file_path.replace("\\", "/") + "\""
    
    def OnTextUpdate(self, event):
        event_name = self.names[event.GetId()]
        if event_name == "output_image_path":
            base_path = "C:/Users/Patrick/Desktop/Anaconda Files/Style Transfer/Output Images/"
            self.script_args[event_name] = "\"" + base_path + event.GetString() + "\""
        else:
            self.script_args[event_name] = event.GetString()

    def OnCheck(self, event):
        if event.GetEventObject().GetValue():
            self.script_args[self.names[event.GetId()]] = ""
        else:
            del self.script_args[self.names[event.GetId()]]
    
    def OnDropdownSelect(self, event):
        if len(event.GetString()) > 0:
            self.script_args[self.names[event.GetId()]] = event.GetString()
    
    def OnMultiSelect(self, event):
        selected = event.GetEventObject().GetCheckedStrings()
        if len(selected) > 0:
            formatted = ""
            for item in selected:
                formatted = formatted + item + " "
            self.script_args[self.names[event.GetId()]] = formatted
        else:
            del self.script_args[self.names[event.GetId()]]
    
    def UnSelectAll(self, event):
        selected = event.GetEventObject().GetSelections()
        for item in selected:
            event.GetEventObject().Deselect(item)
    
    def OnView(self, event):
        wx.MessageBox("Current script call:\n" + build_script_call(self.script_args))
    
    def OnRun(self, event):
        os.system(build_script_call(self.script_args))

def build_script_call(args_dict):
    call_string = "python style_transfer.py"
    for k, v in args_dict.items():
        call_string += " -" + k + " " + v
    return call_string

if __name__ == '__main__':
    app = wx.App()
    frm = ScriptHelperFrame(None, title='NST Script Helper', size=(925, 525))
    frm.Show()
    app.MainLoop()